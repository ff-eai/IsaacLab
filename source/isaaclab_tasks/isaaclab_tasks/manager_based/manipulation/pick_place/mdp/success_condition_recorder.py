# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Recorder for A2 pick-place success sub-conditions and per-step failure labels."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.managers.recorder_manager import RecorderTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.recorder_manager import RecorderTermCfg
from isaaclab.utils import configclass

from .terminations import (
    _env_step_dt,
    classify_a2_failure,
    evaluate_a2_place_conditions,
)

# Termination terms where True means failure; invert when writing failure_reasons
# so True always means the desirable/safe condition is satisfied.
_INVERTED_TERMINATION_TERMS = frozenset({"object_dropping", "time_out"})

_SUCCESS_SUBTERM_KEYS = (
    ("xy_in_tray", "in_xy"),
    ("on_surface_z", "in_z"),
    ("lift_latched", "lift_latched"),
    ("hands_released", "released"),
)


class PickPlaceSuccessConditionRecorder(RecorderTerm):
    """Records per-step success sub-conditions and final failure_reasons snapshot."""

    def record_post_step(self):
        conds = evaluate_a2_place_conditions(
            self._env,
            object_a_cfg=self.cfg.object_a_cfg,
            robot_cfg=self.cfg.robot_cfg,
            xy_threshold=self.cfg.xy_threshold,
            surface_band_low=self.cfg.surface_band_low,
            surface_band_high=self.cfg.surface_band_high,
            table_top_z=self.cfg.table_top_z,
            lift_threshold_m=self.cfg.lift_threshold_m,
            hand_body_names=self.cfg.hand_body_names,
            release_distance_m=self.cfg.release_distance_m,
            grasp_threshold=self.cfg.grasp_threshold,
            lift_hold_time_s=self.cfg.lift_hold_time_s,
            grasp_joint_names=self.cfg.grasp_joint_names,
            gripper_closed_threshold=self.cfg.gripper_closed_threshold,
            spawn_ignore_steps=self.cfg.spawn_ignore_steps,
            knock_z_drop_m=self.cfg.knock_z_drop_m,
            knock_xy_jerk_m=self.cfg.knock_xy_jerk_m,
            knock_displacement_m=self.cfg.knock_displacement_m,
            knock_eef_speed_m=self.cfg.knock_eef_speed_m,
            pick_without_close_z_rise_m=self.cfg.pick_without_close_z_rise_m,
            pick_without_close_eef_rise_m=self.cfg.pick_without_close_eef_rise_m,
            eef_body_name=self.cfg.eef_body_name,
        )
        return "labels", {
            "xy_in_tray": conds["in_xy"],
            "on_surface_z": conds["in_z"],
            "lift_latched": conds["lift_latched"],
            "hands_released": conds["released"],
            "min_hand_dist": conds["min_hand_dist"],
            "object_z": conds["z_can"],
            "xy_dist": conds["xy_dist"],
            "z_offset": conds["z_offset"],
            "hands_near_can": conds["hands_near_can"],
            "gripper_closed": conds["gripper_closed"],
            "can_knocked_now": conds["can_knocked_now"],
            "spawn_can_knocked": conds["spawn_can_knocked"],
            "pick_without_close_now": conds["pick_without_close_now"],
        }

    def record_pre_reset(self, env_ids: Sequence[int] | None):
        conds = evaluate_a2_place_conditions(
            self._env,
            object_a_cfg=self.cfg.object_a_cfg,
            robot_cfg=self.cfg.robot_cfg,
            xy_threshold=self.cfg.xy_threshold,
            surface_band_low=self.cfg.surface_band_low,
            surface_band_high=self.cfg.surface_band_high,
            table_top_z=self.cfg.table_top_z,
            lift_threshold_m=self.cfg.lift_threshold_m,
            hand_body_names=self.cfg.hand_body_names,
            release_distance_m=self.cfg.release_distance_m,
            grasp_threshold=self.cfg.grasp_threshold,
            lift_hold_time_s=self.cfg.lift_hold_time_s,
            grasp_joint_names=self.cfg.grasp_joint_names,
            gripper_closed_threshold=self.cfg.gripper_closed_threshold,
            spawn_ignore_steps=self.cfg.spawn_ignore_steps,
            knock_z_drop_m=self.cfg.knock_z_drop_m,
            knock_xy_jerk_m=self.cfg.knock_xy_jerk_m,
            knock_displacement_m=self.cfg.knock_displacement_m,
            knock_eef_speed_m=self.cfg.knock_eef_speed_m,
            pick_without_close_z_rise_m=self.cfg.pick_without_close_z_rise_m,
            pick_without_close_eef_rise_m=self.cfg.pick_without_close_eef_rise_m,
            eef_body_name=self.cfg.eef_body_name,
        )

        # Compute max episode steps from env config
        max_steps = None
        if hasattr(self._env.cfg, "horizon"):
            max_steps = int(self._env.cfg.horizon)

        env_id_list = list(env_ids if env_ids is not None else range(self._env.num_envs))
        for eid in env_id_list:
            ep = self._env.recorder_manager.get_episode(eid)
            if ep is None:
                continue

            reasons: dict[str, bool] = dict(ep.failure_reasons or {})

            if hasattr(self._env, "termination_manager") and self._env.termination_manager:
                for term_name in self._env.termination_manager.active_terms:
                    term_result = self._env.termination_manager.get_term(term_name)
                    if term_result is None or term_result.numel() == 0:
                        continue
                    val = bool(term_result[eid].item())
                    if term_name in _INVERTED_TERMINATION_TERMS:
                        val = not val
                    reasons[term_name] = val

            for label_key, cond_key in _SUCCESS_SUBTERM_KEYS:
                reasons[label_key] = bool(conds[cond_key][eid].item())

            # --- PI0.7-inspired rich metadata ---
            # Classify failure type and compute quality score
            ep_length = int(self._env.episode_length_buf[eid].item())
            failure_category, quality_score = classify_a2_failure(
                self._env, conds,
                episode_length=ep_length,
                max_episode_steps=max_steps,
                table_top_z=self.cfg.table_top_z,
                lift_threshold_m=self.cfg.lift_threshold_m,
            )
            reasons["failure_category"] = failure_category
            reasons["quality_score"] = quality_score

            # Additional episode metadata
            dt = _env_step_dt(self._env)
            reasons["episode_length_steps"] = ep_length
            reasons["episode_duration_s"] = ep_length * dt
            reasons["max_episode_steps"] = max_steps or -1

            ep.failure_reasons = reasons

        return None, None


@configclass
class PickPlaceSuccessConditionRecorderCfg(RecorderTermCfg):
    """Configuration for pick-place success sub-condition recorder."""

    class_type: type[RecorderTerm] = PickPlaceSuccessConditionRecorder

    object_a_cfg: SceneEntityCfg = SceneEntityCfg("object")
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    xy_threshold: float = 0.10
    surface_band_low: float = 0.00
    surface_band_high: float = 0.10
    table_top_z: float = 1.00
    lift_threshold_m: float = 0.05
    hand_body_names: tuple[str, ...] = ("left_arm_link07", "right_arm_link07")
    release_distance_m: float = 0.15
    grasp_threshold: float = 0.20
    lift_hold_time_s: float = 0.5
    grasp_joint_names: tuple[str, ...] = ("R_thumb_1_joint", "R_index_1_joint")
    gripper_closed_threshold: float = 0.5
    spawn_ignore_steps: int = 5
    knock_z_drop_m: float = 0.008
    knock_xy_jerk_m: float = 0.008
    knock_displacement_m: float = 0.003
    knock_eef_speed_m: float = 0.003
    pick_without_close_z_rise_m: float = 0.004
    pick_without_close_eef_rise_m: float = 0.008
    eef_body_name: str = "right_arm_link07"
