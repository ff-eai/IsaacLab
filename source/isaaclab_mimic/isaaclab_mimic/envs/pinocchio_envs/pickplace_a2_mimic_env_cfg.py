# Copyright (c) 2024-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Mimic subgoal config for A2 place-can-into-tray.

Forked from pickplace_gr1t2_mimic_env_cfg.py. Two-arm subtask configuration
simplified to right-hand-only since can-to-tray is a one-handed task; left hand
is held passively via the idle pose.
"""

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.pick_place.pickplace_recorder_cfg import PickPlaceRecorderManagerCfg
from isaaclab_tasks.manager_based.manipulation.pick_place import mdp
from isaaclab_tasks.manager_based.manipulation.pick_place.pickplace_a2_env_cfg import (
    PickPlaceA2EnvCfg,
    enable_pickplace_a2_cameras,
)


@configclass
class PickPlaceA2MimicEnvCfg(PickPlaceA2EnvCfg, MimicEnvCfg):
    """Mimic config for A2 can-to-tray."""

    recorders = PickPlaceRecorderManagerCfg()
    camera_enabled: bool = False
    """Enable cameras (requires --enable_cameras flag in AppLauncher)."""

    def __post_init__(self):
        super().__post_init__()

        # Use the strict `place_on_tray_with_lift` for generation so the
        # generated dataset enforces all four live-env conditions:
        # xy-in-tray + on-surface-z + lift-latched + hands-released.
        #
        # During mimic generation (waypoint.py:421) the success function is
        # called after *every* env.step() — the lift latch fires correctly.
        # For annotate-replay we patch annotate_demos.py to also tick the
        # success function per-step (otherwise the latch would never fire
        # because annotate only checks success once at end).
        # Loosened success criteria: xy=0.35 (was 0.10), surface_band=[-0.10, 0.05] (was [0.00, 0.10]), grasp=0.15 (was 0.20).
        self.terminations.success = DoneTerm(
            func=mdp.place_on_tray_with_lift,
            params={
                "object_a_cfg": SceneEntityCfg("object"),
                "robot_cfg": SceneEntityCfg("robot"),
                "xy_threshold": 0.35,
                "surface_band_low": -0.10,
                "surface_band_high": 0.05,
                "table_top_z": 1.00,
                "lift_threshold_m": 0.05,
                "hand_body_names": ("left_arm_link07", "right_arm_link07"),
                "release_distance_m": 0.15,
                "grasp_threshold": 0.15,
                "lift_hold_time_s": 0.5,
            },
        )

        self.datagen_config.name = "a2_place_can_tray_D0"
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

        if self.camera_enabled:
            enable_pickplace_a2_cameras(self)

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
