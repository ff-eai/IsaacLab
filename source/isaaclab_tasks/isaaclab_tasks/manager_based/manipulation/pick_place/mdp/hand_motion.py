# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hand motion and gripper closure signals for pick-place failure labeling."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def evaluate_a2_gripper_closed(
    robot: Articulation,
    grasp_joint_names: tuple[str, ...],
    gripper_closed_threshold: float = 0.5,
) -> torch.Tensor:
    """True when all configured grasp-proxy joints exceed the closed threshold."""
    if not grasp_joint_names:
        return torch.zeros(robot.data.joint_pos.shape[0], dtype=torch.bool, device=robot.device)

    joint_ids, _ = robot.find_joints(list(grasp_joint_names))
    if len(joint_ids) == 0:
        return torch.zeros(robot.data.joint_pos.shape[0], dtype=torch.bool, device=robot.device)

    joint_pos = robot.data.joint_pos[:, joint_ids]
    return torch.all(joint_pos > gripper_closed_threshold, dim=1)


def _body_pos_w(robot: Articulation, body_name: str) -> torch.Tensor | None:
    try:
        body_idx = robot.data.body_names.index(body_name)
    except ValueError:
        return None
    return robot.data.body_pos_w[:, body_idx]


def evaluate_a2_hand_motion_step(
    env: ManagerBasedRLEnv,
    robot: Articulation,
    min_hand_dist: torch.Tensor,
    lift_latched: torch.Tensor,
    can_pos: torch.Tensor,
    xy_dist: torch.Tensor,
    grasp_threshold: float,
    grasp_joint_names: tuple[str, ...],
    gripper_closed_threshold: float,
    spawn_ignore_steps: int = 5,
    knock_z_drop_m: float = 0.008,
    knock_xy_jerk_m: float = 0.008,
    knock_displacement_m: float = 0.003,
    knock_eef_speed_m: float = 0.003,
    pick_without_close_z_rise_m: float = 0.004,
    pick_without_close_eef_rise_m: float = 0.008,
    eef_body_name: str = "right_arm_link07",
) -> dict[str, torch.Tensor]:
    """Per-env hand motion signals for one control step."""
    hands_near = min_hand_dist < grasp_threshold
    gripper_closed = evaluate_a2_gripper_closed(robot, grasp_joint_names, gripper_closed_threshold)
    post_spawn = env.episode_length_buf > spawn_ignore_steps

    if not hasattr(env, "_a2_prev_can_pos"):
        env._a2_prev_can_pos = can_pos.clone()
    if not hasattr(env, "_a2_prev_xy_dist"):
        env._a2_prev_xy_dist = xy_dist.clone()
    if not hasattr(env, "_a2_spawn_can_z"):
        env._a2_spawn_can_z = can_pos[:, 2].clone()
    if not hasattr(env, "_a2_prev_eef_pos"):
        eef_pos = _body_pos_w(robot, eef_body_name)
        env._a2_prev_eef_pos = eef_pos.clone() if eef_pos is not None else None

    just_reset = env.episode_length_buf == 0
    if just_reset.any():
        env._a2_spawn_can_z = torch.where(just_reset, can_pos[:, 2], env._a2_spawn_can_z)
        env._a2_prev_can_pos = torch.where(just_reset.unsqueeze(-1), can_pos, env._a2_prev_can_pos)
        env._a2_prev_xy_dist = torch.where(just_reset, xy_dist, env._a2_prev_xy_dist)

    prev_can = env._a2_prev_can_pos
    prev_xy = env._a2_prev_xy_dist

    eef_pos = _body_pos_w(robot, eef_body_name)
    if eef_pos is None:
        eef_speed = torch.zeros(env.num_envs, device=env.device)
        eef_z_rise = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    else:
        prev_eef = env._a2_prev_eef_pos
        if prev_eef is None or just_reset.any():
            prev_eef = eef_pos
        prev_eef = torch.where(just_reset.unsqueeze(-1), eef_pos, prev_eef)
        eef_speed = torch.linalg.vector_norm(eef_pos - prev_eef, dim=1)
        eef_z_rise = (eef_pos[:, 2] - prev_eef[:, 2]) > pick_without_close_eef_rise_m
        env._a2_prev_eef_pos = eef_pos.clone()

    can_delta = torch.linalg.vector_norm(can_pos - prev_can, dim=1)
    z_drop = (can_pos[:, 2] - prev_can[:, 2]) < -knock_z_drop_m
    z_rise = (can_pos[:, 2] - prev_can[:, 2]) > pick_without_close_z_rise_m
    xy_jerk = torch.abs(xy_dist - prev_xy) > knock_xy_jerk_m
    can_displaced = can_delta > knock_displacement_m

    open_near = hands_near & ~gripper_closed
    knocked_contact = (
        post_spawn
        & open_near
        & ~lift_latched
        & (z_drop | xy_jerk | can_displaced | (eef_speed > knock_eef_speed_m))
    )
    early_window = env.episode_length_buf <= (spawn_ignore_steps + 5)
    spawn_settled = (env._a2_spawn_can_z - can_pos[:, 2]) > knock_z_drop_m
    spawn_can_knocked = early_window & ~lift_latched & (z_drop | can_displaced | spawn_settled)

    pick_without_close_now = open_near & (z_rise | eef_z_rise)

    env._a2_prev_can_pos = can_pos.clone()
    env._a2_prev_xy_dist = xy_dist.clone()

    return {
        "hands_near_can": hands_near,
        "gripper_closed": gripper_closed,
        "can_knocked_now": knocked_contact,
        "spawn_can_knocked": spawn_can_knocked,
        "pick_without_close_now": pick_without_close_now,
    }
