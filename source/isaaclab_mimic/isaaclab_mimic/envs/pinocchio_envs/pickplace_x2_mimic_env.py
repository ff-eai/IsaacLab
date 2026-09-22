# Copyright (c) 2024-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Mimic env logic for X2 (OmniHand T2) place-can-into-tray.

Forked from pickplace_a2_mimic_env.py. The only structural difference is the
hand width: A2 drives 6 joints per hand, X2's OmniHand T2 drives 16 per hand
(10 driven + 6 mimic children, see ``_HAND_JOINTS`` in pickplace_x2_env_cfg.py).

Action layout (matches PinkInverseKinematicsAction, which puts the frame-task
poses first in ``target_eef_link_names`` order and the hand joints last in
``hand_joint_names`` order):

    [left_pos(3), left_quat(4), right_pos(3), right_quat(4),
     left_hand_joints(16), right_hand_joints(16)]

Total action_dim = 14 + 32 = 46.
"""

from collections.abc import Sequence

import torch

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv

# Hand joints per side in `_HAND_JOINTS` (left block first, then right).
NUM_HAND_JOINTS_PER_SIDE = 16
# Wrist pose block: 2 arms x (pos 3 + quat 4).
POSE_BLOCK_DIM = 14
LEFT_HAND_SLICE = slice(POSE_BLOCK_DIM, POSE_BLOCK_DIM + NUM_HAND_JOINTS_PER_SIDE)
RIGHT_HAND_SLICE = slice(POSE_BLOCK_DIM + NUM_HAND_JOINTS_PER_SIDE, POSE_BLOCK_DIM + 2 * NUM_HAND_JOINTS_PER_SIDE)

# --- idle_right subtask signal tuning -------------------------------------
# The can's resting height is sampled per episode instead of hardcoded: the
# RoboTwin 071_can origin is not guaranteed to sit at the mesh base, and the
# reset event drops the can from z=1.10 onto a table whose top is at z=1.00, so
# an absolute z threshold is fragile. Baseline is latched during the first few
# steps (after the drop settles), then "lifted" is relative to it.
SETTLE_STEPS = 5  # 5 steps x (decimation 6 / 120 Hz) = 0.25 s
LIFT_MARGIN_M = 0.07  # can must rise this far above its resting height
# Proximity alone cannot tell a grasp from a rest pose on X2: the right wrist sits
# 0.15-0.18 m from the can at the idle pose, overlapping any threshold tight enough
# to mean "holding it". So proximity is only a sanity gate, and the discriminating
# test is finger closure -- the same pair and threshold the env's `place_after_grasp`
# success term uses.
GRASP_PROXIMITY_M = 0.25
GRASP_JOINT_NAMES = ("R_thumb_mcp_joint", "R_index_pip_joint")
GRASP_CLOSED_THRESHOLD = 0.3


class PickPlaceX2MimicEnv(ManagerBasedRLMimicEnv):
    """X2 OmniHand place-can-into-tray Mimic env."""

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        if env_ids is None:
            env_ids = slice(None)
        eef_pos_name = f"{eef_name}_eef_pos"
        eef_quat_name = f"{eef_name}_eef_quat"
        target_wrist_position = self.obs_buf["policy"][eef_pos_name][env_ids]
        target_rot_mat = PoseUtils.matrix_from_quat(self.obs_buf["policy"][eef_quat_name][env_ids])
        return PoseUtils.make_pose(target_wrist_position, target_rot_mat)

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        target_left_pos, left_rot = PoseUtils.unmake_pose(target_eef_pose_dict["left"])
        target_right_pos, right_rot = PoseUtils.unmake_pose(target_eef_pose_dict["right"])
        target_left_quat = PoseUtils.quat_from_matrix(left_rot)
        target_right_quat = PoseUtils.quat_from_matrix(right_rot)

        left_hand_action = gripper_action_dict["left"]
        right_hand_action = gripper_action_dict["right"]

        if action_noise_dict is not None:
            target_left_pos += action_noise_dict["left"] * torch.randn_like(target_left_pos)
            target_right_pos += action_noise_dict["right"] * torch.randn_like(target_right_pos)
            target_left_quat += action_noise_dict["left"] * torch.randn_like(target_left_quat)
            target_right_quat += action_noise_dict["right"] * torch.randn_like(target_right_quat)

        return torch.cat(
            (
                target_left_pos,
                target_left_quat,
                target_right_pos,
                target_right_quat,
                left_hand_action,
                right_hand_action,
            ),
            dim=0,
        )

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        target_poses = {}
        left_pos = action[:, 0:3]
        left_rot_mat = PoseUtils.matrix_from_quat(action[:, 3:7])
        target_poses["left"] = PoseUtils.make_pose(left_pos, left_rot_mat)
        right_pos = action[:, 7:10]
        right_rot_mat = PoseUtils.matrix_from_quat(action[:, 10:14])
        target_poses["right"] = PoseUtils.make_pose(right_pos, right_rot_mat)
        return target_poses

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        # Action layout: [14 wrist poses][16 left hand][16 right hand]
        return {"left": actions[:, LEFT_HAND_SLICE], "right": actions[:, RIGHT_HAND_SLICE]}

    def _grasp_joint_ids(self) -> list[int]:
        """Indices of the OmniHand joints used as the grasp proxy (cached)."""
        if getattr(self, "_cached_grasp_joint_ids", None) is None:
            ids, _ = self.scene["robot"].find_joints(list(GRASP_JOINT_NAMES), preserve_order=True)
            self._cached_grasp_joint_ids = ids
        return self._cached_grasp_joint_ids

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """idle_right = grasp complete.

        True once all three hold: the can has risen ``LIFT_MARGIN_M`` above its
        per-episode resting height, the right-hand thumb and index are flexed past
        ``GRASP_CLOSED_THRESHOLD``, and the right wrist is within
        ``GRASP_PROXIMITY_M`` of the can. Closure is what separates a real grasp
        from the idle pose; the other two stop a closed hand elsewhere in the scene,
        or a can knocked upward, from latching the subtask.
        """
        obs = self.obs_buf["policy"]
        # Both of these are in the env frame, so the distance below is valid for
        # num_envs > 1: `get_eef_pos` subtracts env_origins explicitly, and
        # `root_pos_w` is "asset root position in the environment frame" (it
        # subtracts env_origins too, despite the _w name). Do not re-subtract
        # the origins here. Mimic's own `get_object_poses` uses the same frame
        # (scene state read with is_relative=True).
        object_pos = obs["object_pos"]
        right_eef_pos = obs["right_eef_pos"]
        can_z = object_pos[..., 2]

        # Latch the resting height during the settle window at episode start.
        if getattr(self, "_can_rest_z", None) is None or self._can_rest_z.shape[0] != can_z.shape[0]:
            self._can_rest_z = can_z.clone()
        settling = self.episode_length_buf <= SETTLE_STEPS
        if bool(settling.any()):
            self._can_rest_z = torch.where(settling, can_z, self._can_rest_z)

        can_lifted = can_z > self._can_rest_z + LIFT_MARGIN_M
        hand_close = torch.norm(right_eef_pos - object_pos, dim=-1) < GRASP_PROXIMITY_M
        grasp_joint_pos = self.scene["robot"].data.joint_pos[:, self._grasp_joint_ids()]
        hand_closed = grasp_joint_pos.mean(dim=-1) > GRASP_CLOSED_THRESHOLD
        idle_right = can_lifted & hand_closed & hand_close

        if env_ids is None:
            env_ids = slice(None)
        return {"idle_right": idle_right[env_ids]}
