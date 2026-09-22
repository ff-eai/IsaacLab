# Copyright (c) 2024-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Mimic env logic for A2 place-can-into-tray.

Forked from pickplace_gr1t2_mimic_env.py. Action layout:
  [left_pos(3), left_quat(4), right_pos(3), right_quat(4),
   left_hand_joints(6), right_hand_joints(6)]
Total action_dim = 14 + 12 = 26.
"""

from collections.abc import Sequence

import torch

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv


class PickPlaceA2MimicEnv(ManagerBasedRLMimicEnv):
    """A2 place-can-into-tray Mimic env."""

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

        # DEBUG: log first call for env_id=0
        if not hasattr(self, '_debug_count'):
            self._debug_count = 0
        if self._debug_count < 5:
            print(f"[DEBUG] env_id={env_id} right_pos=({target_right_pos[0]:.4f},{target_right_pos[1]:.4f},{target_right_pos[2]:.4f}) "
                  f"left_hand=({left_hand_action[0]:.4f},...) right_hand=({right_hand_action[0]:.4f},...)")
            self._debug_count += 1

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
        # Action layout: [14 wrist poses][6 left hand][6 right hand]
        return {"left": actions[:, 14:20], "right": actions[:, 20:26]}

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        # idle_right = grasp complete. True once the can is lifted off the table AND
        # the right end-effector is close to the can (so lift isn't from something else).
        # Table top sits ~0.85 m; can z >0.92 means ~7cm lift. Hand within 15cm confirms
        # the gripper is actually the lifter. These thresholds are tuned to the
        # RoboTwin 071_can asset + A2 at pos=(0,0.05,0.93).
        if env_ids is None:
            env_ids = slice(None)
        obs = self.obs_buf["policy"]
        object_pos = obs["object_pos"][env_ids]
        right_eef_pos = obs["right_eef_pos"][env_ids]
        can_lifted = object_pos[..., 2] > 0.92
        dist = torch.norm(right_eef_pos - object_pos, dim=-1)
        hand_close = dist < 0.15
        idle_right = can_lifted & hand_close
        return {"idle_right": idle_right}
