# Copyright (c) 2024-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Mimic subgoal config for X2 (OmniHand T2) place-can-into-tray.

Forked from pickplace_a2_mimic_env_cfg.py. Same scene and same two-subtask
structure (grasp the can, then place it in the tray), right-hand-only since
can-to-tray is one-handed; the left hand is held passively at the idle pose.

Differences from the A2 config:
  * Recorders are X2-specific — the A2 ``PickPlaceRecorderManagerCfg`` hardcodes
    A2 camera names (head_camera / chest_*_camera) and A2 body/joint names
    (``left_arm_link07``, ``R_thumb_1_joint``) that don't exist on X2.
  * ``camera_enabled`` *removes* cameras when False. The X2 env declares its four
    head cameras unconditionally, so without this the env can't start unless
    every run passes ``--enable_cameras`` — expensive during generation.
  * The success term is left as the env's own ``place_after_grasp``. Unlike A2's
    ``place_on_tray_with_lift`` it is stateless, so it behaves identically under
    annotate-replay and generation (no lift-latch that needs annotate_demos.py
    patched to tick the success fn per step), and it already checks an X2 grasp
    via the OmniHand thumb/index joints.
"""

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg, CameraRecorderCfg
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.pick_place.mdp.success_condition_recorder import (
    PickPlaceSuccessConditionRecorderCfg,
)
from isaaclab_tasks.manager_based.manipulation.pick_place.pickplace_x2_env_cfg import (
    LEFT_EEF_LINK,
    RIGHT_EEF_LINK,
    PickPlaceX2EnvCfg,
)

# Scene sensors + the policy obs terms that read them. Both are dropped when
# `camera_enabled` is False (managers skip None-valued entries).
_CAMERA_SENSORS = ("rgbd_head_front", "stereo_head_front", "rgb_head_center", "rgb_head_rear")
_CAMERA_OBS_TERMS = (
    "rgbd_head_front_rgb",
    "rgbd_head_front_depth",
    "stereo_head_front_rgb",
    "rgb_head_center_rgb",
    "rgb_head_rear_rgb",
)

# OmniHand grasp proxy — same pair the X2 env's `place_after_grasp` uses.
_GRASP_JOINT_NAMES = ("R_thumb_mcp_joint", "R_index_pip_joint")


@configclass
class PickPlaceX2RecorderManagerCfg(ActionStateRecorderManagerCfg):
    """Action/state recorder plus pick-place success sub-condition labels, X2 names."""

    record_pickplace_success_conditions = PickPlaceSuccessConditionRecorderCfg(
        # Wrist roll links are the surviving bodies after merge_fixed_joints
        # folded L/R_palm into them.
        hand_body_names=(LEFT_EEF_LINK, RIGHT_EEF_LINK),
        eef_body_name=RIGHT_EEF_LINK,
        grasp_joint_names=_GRASP_JOINT_NAMES,
        gripper_closed_threshold=0.3,  # matches grasp_closed_threshold in the env's success term
        table_top_z=1.00,  # procedural table slab: top surface at z=1.00
    )

    # Camera frames for dataset generation. CameraRecorder skips sensors that
    # aren't in the scene, so these are no-ops when camera_enabled is False.
    record_rgbd_head_front = CameraRecorderCfg(
        sensor_names=["rgbd_head_front"],
        data_types=["rgb", "distance_to_image_plane"],
    )
    record_head_rgb_cams = CameraRecorderCfg(
        sensor_names=["stereo_head_front", "rgb_head_center", "rgb_head_rear"],
        data_types=["rgb"],
    )


@configclass
class PickPlaceX2MimicEnvCfg(PickPlaceX2EnvCfg, MimicEnvCfg):
    """Mimic config for X2 can-to-tray."""

    recorders = PickPlaceX2RecorderManagerCfg()
    camera_enabled: bool = False
    """Keep the X2 head cameras. Requires --enable_cameras in AppLauncher."""

    def __post_init__(self):
        super().__post_init__()

        self.datagen_config.name = "x2_place_can_tray_D0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = True
        self.datagen_config.generation_num_trials = 500  # 1 demo → 500 synthetic
        self.datagen_config.generation_select_src_per_subtask = False
        self.datagen_config.generation_select_src_per_arm = False
        self.datagen_config.generation_relative = False
        self.datagen_config.generation_joint_pos = False
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.max_num_failures = 500
        self.datagen_config.num_demo_to_render = 10
        self.datagen_config.num_fail_demo_to_render = 10
        self.datagen_config.seed = 1

        if not self.camera_enabled:
            for sensor_name in _CAMERA_SENSORS:
                setattr(self.scene, sensor_name, None)
            for obs_term in _CAMERA_OBS_TERMS:
                setattr(self.observations.policy, obs_term, None)

        # Right-arm subtasks: grasp can → place into tray.
        right_subtasks = []
        right_subtasks.append(
            SubTaskConfig(
                object_ref="object",  # the can
                description="grasp the can",
                next_subtask_description="place the can in the tray",
                subtask_term_signal="idle_right",
                first_subtask_start_offset_range=(0, 0),
                subtask_term_offset_range=(0, 10),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.003,
                num_interpolation_steps=10,
                num_fixed_steps=5,
                apply_noise_during_interpolation=False,
            )
        )
        right_subtasks.append(
            SubTaskConfig(
                object_ref="tray",  # the destination
                description="place the can in the tray",
                next_subtask_description="",
                subtask_term_signal=None,  # final subtask
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.003,
                num_interpolation_steps=10,
                num_fixed_steps=5,
                apply_noise_during_interpolation=False,
            )
        )
        self.subtask_configs["right"] = right_subtasks

        # Left arm is passive — one subtask that tracks the can frame for coordination.
        self.subtask_configs["left"] = [
            SubTaskConfig(
                object_ref="object",
                subtask_term_signal=None,
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.003,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        ]
