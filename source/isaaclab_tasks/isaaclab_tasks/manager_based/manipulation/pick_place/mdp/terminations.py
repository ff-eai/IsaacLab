# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from .hand_motion import evaluate_a2_gripper_closed, evaluate_a2_hand_motion_step

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _env_step_dt(env: ManagerBasedRLEnv) -> float:
    """Control-step duration in seconds."""
    return env.cfg.sim.dt * env.cfg.decimation


def _update_a2_lift_latch(
    env: ManagerBasedRLEnv,
    z_can: torch.Tensor,
    min_hand_dist: torch.Tensor,
    table_top_z: float,
    lift_threshold_m: float,
    grasp_threshold: float,
    lift_hold_time_s: float = 0.5,
) -> torch.Tensor:
    """Update and return the per-env lift latch (object grasped, lifted, and held).

    The lift latch only sets when:
      1. A hand was within ``grasp_threshold`` of the can earlier this episode.
      2. The can z stays above ``table_top_z + lift_threshold_m`` continuously for
         at least ``lift_hold_time_s`` (default 0.5 s of control steps).
    """
    if not hasattr(env, "_a2_was_near_can"):
        env._a2_was_near_can = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if not hasattr(env, "_a2_was_lifted"):
        env._a2_was_lifted = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if not hasattr(env, "_a2_lift_sustain_steps"):
        env._a2_lift_sustain_steps = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)

    just_reset = env.episode_length_buf == 0
    env._a2_was_near_can = env._a2_was_near_can & ~just_reset
    env._a2_was_lifted = env._a2_was_lifted & ~just_reset
    env._a2_lift_sustain_steps = torch.where(
        just_reset,
        torch.zeros_like(env._a2_lift_sustain_steps),
        env._a2_lift_sustain_steps,
    )

    near_now = min_hand_dist < grasp_threshold
    env._a2_was_near_can = env._a2_was_near_can | near_now

    lift_height = table_top_z + lift_threshold_m
    lifted_now = z_can > lift_height
    eligible = env._a2_was_near_can
    sustaining = lifted_now & eligible

    env._a2_lift_sustain_steps = torch.where(
        sustaining,
        env._a2_lift_sustain_steps + 1,
        torch.zeros_like(env._a2_lift_sustain_steps),
    )

    required_steps = max(1, int(round(lift_hold_time_s / _env_step_dt(env))))
    sustained = env._a2_lift_sustain_steps >= required_steps
    env._a2_was_lifted = env._a2_was_lifted | sustained
    return env._a2_was_lifted


def evaluate_a2_place_conditions(
    env: ManagerBasedRLEnv,
    object_a_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    xy_threshold: float = 0.10,
    surface_band_low: float = 0.00,
    surface_band_high: float = 0.10,
    table_top_z: float = 1.00,
    lift_threshold_m: float = 0.05,
    hand_body_names: tuple[str, ...] = ("left_arm_link07", "right_arm_link07"),
    release_distance_m: float = 0.15,
    grasp_threshold: float = 0.15,
    lift_hold_time_s: float = 0.5,
    grasp_joint_names: tuple[str, ...] = ("R_thumb_1_joint", "R_index_1_joint"),
    gripper_closed_threshold: float = 0.5,
    spawn_ignore_steps: int = 5,
    knock_z_drop_m: float = 0.008,
    knock_xy_jerk_m: float = 0.008,
    knock_displacement_m: float = 0.003,
    knock_eef_speed_m: float = 0.003,
    pick_without_close_z_rise_m: float = 0.004,
    pick_without_close_eef_rise_m: float = 0.008,
    eef_body_name: str = "right_arm_link07",
) -> dict[str, torch.Tensor]:
    """Evaluate pick-place success sub-conditions for labeling (does not terminate).

    Returns tensors shaped (num_envs,) with keys:
      in_xy, in_z, lift_latched, released, min_hand_dist, z_can, xy_dist, z_offset

    ``lift_latched`` requires hand proximity, then the can z staying above the table
    surface band for ``lift_hold_time_s``. ``released`` is True only after
    lift_latched and both hand bodies are farther than ``release_distance_m``.
    """
    object_a: RigidObject = env.scene[object_a_cfg.name]
    object_b: RigidObject = env.scene["tray"]
    robot: Articulation = env.scene[robot_cfg.name]

    z_can = object_a.data.root_pos_w[:, 2]

    pos_diff = object_a.data.root_pos_w - object_b.data.root_pos_w
    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)
    if hasattr(env, "_tray_top_z"):
        ref_top = env._tray_top_z
    else:
        ref_top = object_b.data.root_pos_w[:, 2] + 0.05
    z_offset = z_can - ref_top
    in_xy = xy_dist < xy_threshold
    in_z = (z_offset >= surface_band_low) & (z_offset <= surface_band_high)

    can_pos = object_a.data.root_pos_w
    min_hand_dist = torch.full(
        (env.num_envs,), float("inf"), device=env.device, dtype=torch.float32,
    )
    for body_name in hand_body_names:
        try:
            body_idx = robot.data.body_names.index(body_name)
            hand_pos = robot.data.body_pos_w[:, body_idx]
            d = torch.linalg.vector_norm(can_pos - hand_pos, dim=1)
            min_hand_dist = torch.minimum(min_hand_dist, d)
        except ValueError:
            continue

    lift_latched = _update_a2_lift_latch(
        env, z_can, min_hand_dist, table_top_z, lift_threshold_m, grasp_threshold, lift_hold_time_s,
    )
    hands_far = min_hand_dist > release_distance_m
    # Require prior lift (grasp) before counting as released — idle hands far from
    # the can at episode start must not read as hands_released.
    released = lift_latched & hands_far

    hand_motion = evaluate_a2_hand_motion_step(
        env,
        robot,
        min_hand_dist,
        lift_latched,
        can_pos,
        xy_dist,
        grasp_threshold,
        grasp_joint_names,
        gripper_closed_threshold,
        spawn_ignore_steps=spawn_ignore_steps,
        knock_z_drop_m=knock_z_drop_m,
        knock_xy_jerk_m=knock_xy_jerk_m,
        knock_displacement_m=knock_displacement_m,
        knock_eef_speed_m=knock_eef_speed_m,
        pick_without_close_z_rise_m=pick_without_close_z_rise_m,
        pick_without_close_eef_rise_m=pick_without_close_eef_rise_m,
        eef_body_name=eef_body_name,
    )

    return {
        "in_xy": in_xy,
        "in_z": in_z,
        "lift_latched": lift_latched,
        "released": released,
        "min_hand_dist": min_hand_dist,
        "z_can": z_can,
        "xy_dist": xy_dist,
        "z_offset": z_offset,
        "hands_near_can": hand_motion["hands_near_can"],
        "gripper_closed": hand_motion["gripper_closed"],
        "can_knocked_now": hand_motion["can_knocked_now"],
        "spawn_can_knocked": hand_motion["spawn_can_knocked"],
        "pick_without_close_now": hand_motion["pick_without_close_now"],
    }


def check_xy_in_tray(
    env: ManagerBasedRLEnv,
    object_a_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    xy_threshold: float = 0.10,
    surface_band_low: float = 0.00,
    surface_band_high: float = 0.10,
    table_top_z: float = 1.00,
    lift_threshold_m: float = 0.05,
    hand_body_names: tuple[str, ...] = ("left_arm_link07", "right_arm_link07"),
    release_distance_m: float = 0.15,
) -> torch.Tensor:
    """True when the object is within the tray xy footprint."""
    return evaluate_a2_place_conditions(
        env,
        object_a_cfg=object_a_cfg,
        robot_cfg=robot_cfg,
        xy_threshold=xy_threshold,
        surface_band_low=surface_band_low,
        surface_band_high=surface_band_high,
        table_top_z=table_top_z,
        lift_threshold_m=lift_threshold_m,
        hand_body_names=hand_body_names,
        release_distance_m=release_distance_m,
    )["in_xy"]


def check_on_surface_z(
    env: ManagerBasedRLEnv,
    object_a_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    xy_threshold: float = 0.10,
    surface_band_low: float = 0.00,
    surface_band_high: float = 0.10,
    table_top_z: float = 1.00,
    lift_threshold_m: float = 0.05,
    hand_body_names: tuple[str, ...] = ("left_arm_link07", "right_arm_link07"),
    release_distance_m: float = 0.15,
) -> torch.Tensor:
    """True when the object height is within the tray surface band."""
    return evaluate_a2_place_conditions(
        env,
        object_a_cfg=object_a_cfg,
        robot_cfg=robot_cfg,
        xy_threshold=xy_threshold,
        surface_band_low=surface_band_low,
        surface_band_high=surface_band_high,
        table_top_z=table_top_z,
        lift_threshold_m=lift_threshold_m,
        hand_body_names=hand_body_names,
        release_distance_m=release_distance_m,
    )["in_z"]


def check_lift_latched(
    env: ManagerBasedRLEnv,
    object_a_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    xy_threshold: float = 0.10,
    surface_band_low: float = 0.00,
    surface_band_high: float = 0.10,
    table_top_z: float = 1.00,
    lift_threshold_m: float = 0.05,
    hand_body_names: tuple[str, ...] = ("left_arm_link07", "right_arm_link07"),
    release_distance_m: float = 0.15,
) -> torch.Tensor:
    """True when the object was lifted above the table at some point this episode."""
    return evaluate_a2_place_conditions(
        env,
        object_a_cfg=object_a_cfg,
        robot_cfg=robot_cfg,
        xy_threshold=xy_threshold,
        surface_band_low=surface_band_low,
        surface_band_high=surface_band_high,
        table_top_z=table_top_z,
        lift_threshold_m=lift_threshold_m,
        hand_body_names=hand_body_names,
        release_distance_m=release_distance_m,
    )["lift_latched"]


def check_hands_released(
    env: ManagerBasedRLEnv,
    object_a_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    xy_threshold: float = 0.10,
    surface_band_low: float = 0.00,
    surface_band_high: float = 0.10,
    table_top_z: float = 1.00,
    lift_threshold_m: float = 0.05,
    hand_body_names: tuple[str, ...] = ("left_arm_link07", "right_arm_link07"),
    release_distance_m: float = 0.15,
) -> torch.Tensor:
    """True when the can was lifted (grasped) and hands are now clear of the object."""
    return evaluate_a2_place_conditions(
        env,
        object_a_cfg=object_a_cfg,
        robot_cfg=robot_cfg,
        xy_threshold=xy_threshold,
        surface_band_low=surface_band_low,
        surface_band_high=surface_band_high,
        table_top_z=table_top_z,
        lift_threshold_m=lift_threshold_m,
        hand_body_names=hand_body_names,
        release_distance_m=release_distance_m,
    )["released"]


def place_on_tray_with_lift(
    env: ManagerBasedRLEnv,
    object_a_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    xy_threshold: float = 0.10,
    surface_band_low: float = 0.00,
    surface_band_high: float = 0.10,
    table_top_z: float = 1.00,
    lift_threshold_m: float = 0.05,
    hand_body_names: tuple[str, ...] = ("left_arm_link07", "right_arm_link07"),
    release_distance_m: float = 0.15,
    grasp_threshold: float = 0.15,
    lift_hold_time_s: float = 0.5,
) -> torch.Tensor:
    """Live-env success check (record_demos): position-on-tray + lift + release.

    Success requires four conditions held simultaneously:
      1. xy distance(can_root, tray_root) < xy_threshold
      2. z_can ∈ [tray_top_z + surface_band_low, tray_top_z + surface_band_high]
         (where `tray_top_z` is stashed by `_compute_tray_top_z` at startup;
         fallback: tray_root + 5cm)
      3. Lift latch: hand within ``grasp_threshold``, then can z above the table
         surface (``table_top_z + lift_threshold_m``) sustained for ``lift_hold_time_s``.
      4. Release: lift latched and both hand bodies at least ``release_distance_m``
         from the can right now."""
    conds = evaluate_a2_place_conditions(
        env,
        object_a_cfg=object_a_cfg,
        robot_cfg=robot_cfg,
        xy_threshold=xy_threshold,
        surface_band_low=surface_band_low,
        surface_band_high=surface_band_high,
        table_top_z=table_top_z,
        lift_threshold_m=lift_threshold_m,
        hand_body_names=hand_body_names,
        release_distance_m=release_distance_m,
        grasp_threshold=grasp_threshold,
        lift_hold_time_s=lift_hold_time_s,
    )
    in_xy = conds["in_xy"]
    in_z = conds["in_z"]
    released = conds["released"]
    xy_dist = conds["xy_dist"]
    z_offset = conds["z_offset"]
    z_can = conds["z_can"]
    min_hand_dist = conds["min_hand_dist"]

    success = in_xy & in_z & conds["lift_latched"] & released

    if not hasattr(env, "_success_dbg_step"):
        env._success_dbg_step = 0
    env._success_dbg_step += 1
    if env._success_dbg_step % 30 == 0:
        i = 0
        print(
            f"[Success_DBG] in_xy={bool(in_xy[i].item())} in_z={bool(in_z[i].item())} "
            f"lifted={bool(conds['lift_latched'][i].item())} released={bool(released[i].item())} "
            f"-> success={bool(success[i].item())}  "
            f"| xy_dist={float(xy_dist[i].item()):.3f}/{xy_threshold:.2f} "
            f"z_offset={float(z_offset[i].item()):.3f} (band [{surface_band_low:.2f},{surface_band_high:.2f}]) "
            f"can_z={float(z_can[i].item()):.3f} (lift>{table_top_z + lift_threshold_m:.2f}) "
            f"min_hand_dist={float(min_hand_dist[i].item()):.3f} (grasp<{grasp_threshold:.2f}, release>{release_distance_m:.2f})"
        )

    return success


def object_dropped_after_lift(
    env: ManagerBasedRLEnv,
    minimum_height: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    table_top_z: float = 1.00,
    lift_threshold_m: float = 0.05,
    grasp_threshold: float = 0.15,
    lift_hold_time_s: float = 0.5,
    hand_body_names: tuple[str, ...] = ("left_arm_link07", "right_arm_link07"),
    release_distance_m: float = 0.15,
) -> torch.Tensor:
    """Terminate when the object falls below ``minimum_height`` after it was lifted.

    Drop is ignored until the lift latch has fired (hand grasp + sustained lift).
    """
    conds = evaluate_a2_place_conditions(
        env,
        object_a_cfg=asset_cfg,
        robot_cfg=robot_cfg,
        table_top_z=table_top_z,
        lift_threshold_m=lift_threshold_m,
        grasp_threshold=grasp_threshold,
        lift_hold_time_s=lift_hold_time_s,
        hand_body_names=hand_body_names,
        release_distance_m=release_distance_m,
    )
    dropped_now = conds["z_can"] < minimum_height
    return dropped_now & conds["lift_latched"]


def place_on_tray_surface(
    env: ManagerBasedRLEnv,
    object_a_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    xy_threshold: float = 0.18,
    surface_band_low: float = 0.00,
    surface_band_high: float = 0.10,
) -> torch.Tensor:
    """Stateless success check using the tray's actual top-surface z.

    Requires `env._tray_top_z` (per-env tensor) to be set by the
    `_compute_tray_top_z` startup event. Success when:
      * xy distance(can_root, tray_root) < xy_threshold
      * z_can ∈ [tray_top_z + surface_band_low, tray_top_z + surface_band_high]
        (i.e. the can's root is sitting on or just above the tray's top surface)

    surface_band_low=0.0, surface_band_high=0.10 = "can within 10cm above the
    tray's top, not below it". Adjust if the can's root frame is offset from
    its bottom by more than a few cm."""
    object_a: RigidObject = env.scene[object_a_cfg.name]
    object_b: RigidObject = env.scene["tray"]
    pos_diff = object_a.data.root_pos_w - object_b.data.root_pos_w
    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)
    if not hasattr(env, "_tray_top_z"):
        # fallback to tray-root + ~5cm (matches typical 008_tray rim height)
        ref_top = object_b.data.root_pos_w[:, 2] + 0.05
    else:
        ref_top = env._tray_top_z
    z_can = object_a.data.root_pos_w[:, 2]
    z_offset = z_can - ref_top
    in_xy = xy_dist < xy_threshold
    in_z = (z_offset >= surface_band_low) & (z_offset <= surface_band_high)
    return in_xy & in_z


def place_simple_in_tray(
    env: ManagerBasedRLEnv,
    object_a_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    object_b_cfg: SceneEntityCfg = SceneEntityCfg("tray"),
    xy_threshold: float = 0.18,
    height_diff: float = 0.0,
    height_threshold: float = 100.0,
) -> torch.Tensor:
    """Stateless variant of place_after_grasp — checks only that object_a is
    within (xy_threshold, height_band) of object_b. No latch, no gripper check.

    Intended for the Mimic annotate flow, where the script detaches
    terminations.success so the latch in place_after_grasp never fires. The
    upstream record_demos pipeline already filters to successful demos via
    EXPORT_SUCCEEDED_ONLY, so the mimic source dataset can be trusted as
    pre-validated. This term then just confirms the demo's final state still
    has the can in the tray xy-bound when annotation evaluates the end of the
    replay."""
    object_a: RigidObject = env.scene[object_a_cfg.name]
    object_b: RigidObject = env.scene[object_b_cfg.name]
    pos_diff = object_a.data.root_pos_w - object_b.data.root_pos_w
    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)
    height_dist = torch.abs(pos_diff[:, 2])
    return (xy_dist < xy_threshold) & ((height_dist - height_diff) < height_threshold)


def place_after_grasp(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_a_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    object_b_cfg: SceneEntityCfg = SceneEntityCfg("tray"),
    xy_threshold: float = 0.10,
    height_threshold: float = 0.05,
    height_diff: float = 0.05,
    grasp_joint_names: tuple[str, ...] = ("R_thumb_1_joint", "R_index_1_joint"),
    grasp_closed_threshold: float = 0.3,
    grasp_proximity: float = 0.15,
    hand_body_name: str = "right_arm_link07",
    grasp_joint_names_alt: tuple[str, ...] | None = None,
    hand_body_name_alt: str | None = None,
) -> torch.Tensor:
    """Success when object_a is in object_b AND was previously grasped AND now released.

    Latches a per-env ``_a2_was_grasped`` flag on ``env`` the first step the gripper
    joints exceed ``grasp_closed_threshold`` while the object is within
    ``grasp_proximity`` of ``hand_body_name``. Fires success only when that flag is
    set, the object is positioned in the target, and the gripper joints are open.
    The flag is cleared for any env whose ``episode_length_buf == 0`` (just reset).

    If both ``grasp_joint_names_alt`` and ``hand_body_name_alt`` are provided, the
    same grasp/release latch is tracked separately for the alt hand under
    ``_a2_was_grasped_alt`` and the result is OR'd: success fires if EITHER hand
    completed the grasp-then-place sequence (with the same hand being open at
    release for at least one side).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    object_a: RigidObject = env.scene[object_a_cfg.name]
    object_b: RigidObject = env.scene[object_b_cfg.name]

    if not hasattr(env, "_a2_was_grasped"):
        env._a2_was_grasped = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if not hasattr(env, "_a2_was_grasped_alt"):
        env._a2_was_grasped_alt = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    just_reset = env.episode_length_buf == 0
    env._a2_was_grasped = env._a2_was_grasped & ~just_reset
    env._a2_was_grasped_alt = env._a2_was_grasped_alt & ~just_reset

    pos_diff = object_a.data.root_pos_w - object_b.data.root_pos_w
    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)
    height_dist = torch.abs(pos_diff[:, 2])
    in_target = (xy_dist < xy_threshold) & ((height_dist - height_diff) < height_threshold)

    def _eval_side(joint_names: tuple[str, ...], body_name: str):
        joint_ids, _ = robot.find_joints(list(joint_names))
        joint_pos = robot.data.joint_pos[:, joint_ids]
        closed = torch.all(joint_pos > grasp_closed_threshold, dim=1)
        opened = torch.all(joint_pos < grasp_closed_threshold, dim=1)
        body_idx = robot.data.body_names.index(body_name)
        body_pos_w = robot.data.body_pos_w[:, body_idx]
        obj_to_hand = torch.linalg.vector_norm(
            object_a.data.root_pos_w - body_pos_w, dim=1
        )
        near = obj_to_hand < grasp_proximity
        return closed, opened, obj_to_hand, near, joint_pos

    closed_p, opened_p, dist_p, near_p, jp_p = _eval_side(grasp_joint_names, hand_body_name)
    env._a2_was_grasped = env._a2_was_grasped | (closed_p & near_p)
    success_primary = env._a2_was_grasped & in_target & opened_p

    use_alt = grasp_joint_names_alt is not None and hand_body_name_alt is not None
    if use_alt:
        closed_a, opened_a, dist_a, near_a, jp_a = _eval_side(
            grasp_joint_names_alt, hand_body_name_alt
        )
        env._a2_was_grasped_alt = env._a2_was_grasped_alt | (closed_a & near_a)
        success_alt = env._a2_was_grasped_alt & in_target & opened_a
        success = success_primary | success_alt
    else:
        success = success_primary

    # Rate-limited diagnostic for env 0.
    if not hasattr(env, "_success_dbg_step"):
        env._success_dbg_step = 0
    env._success_dbg_step += 1
    if env._success_dbg_step % 30 == 0:
        i = 0
        joint_str_p = " ".join(
            f"{n}={float(jp_p[i, k].item()):.2f}"
            for k, n in enumerate(grasp_joint_names)
        )
        line = (
            f"[Success_DBG] R_grasped={bool(env._a2_was_grasped[i].item())} "
            f"in_target={bool(in_target[i].item())} R_open={bool(opened_p[i].item())} "
            f"-> success={bool(success[i].item())}  "
            f"| xy_dist={float(xy_dist[i].item()):.3f} "
            f"height_dist={float(height_dist[i].item()):.3f} "
            f"R_obj_to_hand={float(dist_p[i].item()):.3f} | {joint_str_p}"
        )
        if use_alt:
            joint_str_a = " ".join(
                f"{n}={float(jp_a[i, k].item()):.2f}"
                for k, n in enumerate(grasp_joint_names_alt)
            )
            line += (
                f"  || L_grasped={bool(env._a2_was_grasped_alt[i].item())} "
                f"L_open={bool(opened_a[i].item())} "
                f"L_obj_to_hand={float(dist_a[i].item()):.3f} | {joint_str_a}"
            )
        print(line)

    return success


def task_done_pick_place(
    env: ManagerBasedRLEnv,
    task_link_name: str = "",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    right_wrist_max_x: float = 0.26,
    min_x: float = 0.40,
    max_x: float = 0.85,
    min_y: float = 0.35,
    max_y: float = 0.60,
    max_height: float = 1.10,
    min_vel: float = 0.20,
) -> torch.Tensor:
    """Determine if the object placement task is complete.

    This function checks whether all success conditions for the task have been met:
    1. object is within the target x/y range
    2. object is below a minimum height
    3. object velocity is below threshold
    4. Right robot wrist is retracted back towards body (past a given x pos threshold)

    Args:
        env: The RL environment instance.
        object_cfg: Configuration for the object entity.
        right_wrist_max_x: Maximum x position of the right wrist for task completion.
        min_x: Minimum x position of the object for task completion.
        max_x: Maximum x position of the object for task completion.
        min_y: Minimum y position of the object for task completion.
        max_y: Maximum y position of the object for task completion.
        max_height: Maximum height (z position) of the object for task completion.
        min_vel: Minimum velocity magnitude of the object for task completion.

    Returns:
        Boolean tensor indicating which environments have completed the task.
    """
    if task_link_name == "":
        raise ValueError("task_link_name must be provided to task_done_pick_place")

    # Get object entity from the scene
    object: RigidObject = env.scene[object_cfg.name]

    # Extract wheel position relative to environment origin
    object_x = object.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    object_y = object.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
    object_height = object.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    object_vel = torch.abs(object.data.root_vel_w)

    # Get right wrist position relative to environment origin
    robot_body_pos_w = env.scene["robot"].data.body_pos_w
    right_eef_idx = env.scene["robot"].data.body_names.index(task_link_name)
    right_wrist_x = robot_body_pos_w[:, right_eef_idx, 0] - env.scene.env_origins[:, 0]

    # Check all success conditions and combine with logical AND
    done = object_x < max_x
    done = torch.logical_and(done, object_x > min_x)
    done = torch.logical_and(done, object_y < max_y)
    done = torch.logical_and(done, object_y > min_y)
    done = torch.logical_and(done, object_height < max_height)
    done = torch.logical_and(done, right_wrist_x < right_wrist_max_x)
    done = torch.logical_and(done, object_vel[:, 0] < min_vel)
    done = torch.logical_and(done, object_vel[:, 1] < min_vel)
    done = torch.logical_and(done, object_vel[:, 2] < min_vel)

    return done


def task_done_nut_pour(
    env: ManagerBasedRLEnv,
    sorting_scale_cfg: SceneEntityCfg = SceneEntityCfg("sorting_scale"),
    sorting_bowl_cfg: SceneEntityCfg = SceneEntityCfg("sorting_bowl"),
    sorting_beaker_cfg: SceneEntityCfg = SceneEntityCfg("sorting_beaker"),
    factory_nut_cfg: SceneEntityCfg = SceneEntityCfg("factory_nut"),
    sorting_bin_cfg: SceneEntityCfg = SceneEntityCfg("black_sorting_bin"),
    max_bowl_to_scale_x: float = 0.055,
    max_bowl_to_scale_y: float = 0.055,
    max_bowl_to_scale_z: float = 0.025,
    max_nut_to_bowl_x: float = 0.050,
    max_nut_to_bowl_y: float = 0.050,
    max_nut_to_bowl_z: float = 0.019,
    max_beaker_to_bin_x: float = 0.08,
    max_beaker_to_bin_y: float = 0.12,
    max_beaker_to_bin_z: float = 0.07,
) -> torch.Tensor:
    """Determine if the nut pouring task is complete.

    This function checks whether all success conditions for the task have been met:
    1. The factory nut is in the sorting bowl
    2. The sorting beaker is in the sorting bin
    3. The sorting bowl is placed on the sorting scale

    Args:
        env: The RL environment instance.
        sorting_scale_cfg: Configuration for the sorting scale entity.
        sorting_bowl_cfg: Configuration for the sorting bowl entity.
        sorting_beaker_cfg: Configuration for the sorting beaker entity.
        factory_nut_cfg: Configuration for the factory nut entity.
        sorting_bin_cfg: Configuration for the sorting bin entity.
        max_bowl_to_scale_x: Maximum x position of the sorting bowl relative to the sorting scale for task completion.
        max_bowl_to_scale_y: Maximum y position of the sorting bowl relative to the sorting scale for task completion.
        max_bowl_to_scale_z: Maximum z position of the sorting bowl relative to the sorting scale for task completion.
        max_nut_to_bowl_x: Maximum x position of the factory nut relative to the sorting bowl for task completion.
        max_nut_to_bowl_y: Maximum y position of the factory nut relative to the sorting bowl for task completion.
        max_nut_to_bowl_z: Maximum z position of the factory nut relative to the sorting bowl for task completion.
        max_beaker_to_bin_x: Maximum x position of the sorting beaker relative to the sorting bin for task completion.
        max_beaker_to_bin_y: Maximum y position of the sorting beaker relative to the sorting bin for task completion.
        max_beaker_to_bin_z: Maximum z position of the sorting beaker relative to the sorting bin for task completion.

    Returns:
        Boolean tensor indicating which environments have completed the task.
    """
    # Get object entities from the scene
    sorting_scale: RigidObject = env.scene[sorting_scale_cfg.name]
    sorting_bowl: RigidObject = env.scene[sorting_bowl_cfg.name]
    factory_nut: RigidObject = env.scene[factory_nut_cfg.name]
    sorting_beaker: RigidObject = env.scene[sorting_beaker_cfg.name]
    sorting_bin: RigidObject = env.scene[sorting_bin_cfg.name]

    # Get positions relative to environment origin
    scale_pos = sorting_scale.data.root_pos_w - env.scene.env_origins
    bowl_pos = sorting_bowl.data.root_pos_w - env.scene.env_origins
    sorting_beaker_pos = sorting_beaker.data.root_pos_w - env.scene.env_origins
    nut_pos = factory_nut.data.root_pos_w - env.scene.env_origins
    bin_pos = sorting_bin.data.root_pos_w - env.scene.env_origins

    # nut to bowl
    nut_to_bowl_x = torch.abs(nut_pos[:, 0] - bowl_pos[:, 0])
    nut_to_bowl_y = torch.abs(nut_pos[:, 1] - bowl_pos[:, 1])
    nut_to_bowl_z = nut_pos[:, 2] - bowl_pos[:, 2]

    # bowl to scale
    bowl_to_scale_x = torch.abs(bowl_pos[:, 0] - scale_pos[:, 0])
    bowl_to_scale_y = torch.abs(bowl_pos[:, 1] - scale_pos[:, 1])
    bowl_to_scale_z = bowl_pos[:, 2] - scale_pos[:, 2]

    # beaker to bin
    beaker_to_bin_x = torch.abs(sorting_beaker_pos[:, 0] - bin_pos[:, 0])
    beaker_to_bin_y = torch.abs(sorting_beaker_pos[:, 1] - bin_pos[:, 1])
    beaker_to_bin_z = sorting_beaker_pos[:, 2] - bin_pos[:, 2]

    done = nut_to_bowl_x < max_nut_to_bowl_x
    done = torch.logical_and(done, nut_to_bowl_y < max_nut_to_bowl_y)
    done = torch.logical_and(done, nut_to_bowl_z < max_nut_to_bowl_z)
    done = torch.logical_and(done, bowl_to_scale_x < max_bowl_to_scale_x)
    done = torch.logical_and(done, bowl_to_scale_y < max_bowl_to_scale_y)
    done = torch.logical_and(done, bowl_to_scale_z < max_bowl_to_scale_z)
    done = torch.logical_and(done, beaker_to_bin_x < max_beaker_to_bin_x)
    done = torch.logical_and(done, beaker_to_bin_y < max_beaker_to_bin_y)
    done = torch.logical_and(done, beaker_to_bin_z < max_beaker_to_bin_z)

    return done


def task_done_exhaust_pipe(
    env: ManagerBasedRLEnv,
    blue_exhaust_pipe_cfg: SceneEntityCfg = SceneEntityCfg("blue_exhaust_pipe"),
    blue_sorting_bin_cfg: SceneEntityCfg = SceneEntityCfg("blue_sorting_bin"),
    max_blue_exhaust_to_bin_x: float = 0.085,
    max_blue_exhaust_to_bin_y: float = 0.200,
    min_blue_exhaust_to_bin_y: float = -0.090,
    max_blue_exhaust_to_bin_z: float = 0.070,
) -> torch.Tensor:
    """Determine if the exhaust pipe task is complete.

    This function checks whether all success conditions for the task have been met:
    1. The blue exhaust pipe is placed in the correct position

    Args:
        env: The RL environment instance.
        blue_exhaust_pipe_cfg: Configuration for the blue exhaust pipe entity.
        blue_sorting_bin_cfg: Configuration for the blue sorting bin entity.
        max_blue_exhaust_to_bin_x: Maximum x position of the blue exhaust pipe
            relative to the blue sorting bin for task completion.
        max_blue_exhaust_to_bin_y: Maximum y position of the blue exhaust pipe
            relative to the blue sorting bin for task completion.
        max_blue_exhaust_to_bin_z: Maximum z position of the blue exhaust pipe
            relative to the blue sorting bin for task completion.

    Returns:
        Boolean tensor indicating which environments have completed the task.
    """
    # Get object entities from the scene
    blue_exhaust_pipe: RigidObject = env.scene[blue_exhaust_pipe_cfg.name]
    blue_sorting_bin: RigidObject = env.scene[blue_sorting_bin_cfg.name]

    # Get positions relative to environment origin
    blue_exhaust_pipe_pos = blue_exhaust_pipe.data.root_pos_w - env.scene.env_origins
    blue_sorting_bin_pos = blue_sorting_bin.data.root_pos_w - env.scene.env_origins

    # blue exhaust to bin
    blue_exhaust_to_bin_x = torch.abs(blue_exhaust_pipe_pos[:, 0] - blue_sorting_bin_pos[:, 0])
    blue_exhaust_to_bin_y = blue_exhaust_pipe_pos[:, 1] - blue_sorting_bin_pos[:, 1]
    blue_exhaust_to_bin_z = blue_exhaust_pipe_pos[:, 2] - blue_sorting_bin_pos[:, 2]

    done = blue_exhaust_to_bin_x < max_blue_exhaust_to_bin_x
    done = torch.logical_and(done, blue_exhaust_to_bin_y < max_blue_exhaust_to_bin_y)
    done = torch.logical_and(done, blue_exhaust_to_bin_y > min_blue_exhaust_to_bin_y)
    done = torch.logical_and(done, blue_exhaust_to_bin_z < max_blue_exhaust_to_bin_z)

    return done


def classify_a2_failure(
    env: ManagerBasedRLEnv,
    sub_conds: dict[str, torch.Tensor],
    episode_length: int | None = None,
    max_episode_steps: int | None = None,
    table_top_z: float = 1.00,
    lift_threshold_m: float = 0.05,
) -> tuple[str, float]:
    """Classify an A2 pick-place episode's outcome and compute a quality score.

    Returns (failure_category, quality_score) where quality_score in [0.0, 1.0].

    Failure categories (mutually exclusive, first-match wins):
      - ``"success"``                  -- all four sub-conditions met
      - ``"knocked_can_early"``        -- can was knocked during spawn phase
      - ``"knocked_can_during_task"``  -- can knocked after spawn
      - ``"never_grasped"``            -- hands never came near the can
      - ``"grasp_without_lift"``       -- can lifted while gripper was still open
      - ``"dropped_after_lift"``       -- lift latched, then can fell
      - ``"placement_missed"``         -- lift+release ok, but xy/z placement wrong
      - ``"timeout_unengaged"``        -- timed out with minimal activity
      - ``"unknown_failure"``          -- fallback

    Args:
        env: The environment instance (used to read can height).
        sub_conds: Dict from evaluate_a2_place_conditions (per-env tensors).
        episode_length: Current step count for env 0 (or None to read from env).
        max_episode_steps: Maximum episode length (or None to infer).
        table_top_z: Table surface height.
        lift_threshold_m: Minimum can rise above table to count as lift.

    Returns:
        Tuple[str, float]: (failure_category, quality_score) for env 0.
    """
    in_xy = bool(sub_conds["in_xy"][0].item())
    in_z = bool(sub_conds["in_z"][0].item())
    lift_latched = bool(sub_conds["lift_latched"][0].item())
    released = bool(sub_conds["released"][0].item())
    hands_near_can = bool(sub_conds["hands_near_can"][0].item())
    can_knocked_now = bool(sub_conds["can_knocked_now"][0].item())
    spawn_can_knocked = bool(sub_conds["spawn_can_knocked"][0].item())
    pick_without_close = bool(sub_conds["pick_without_close_now"][0].item())
    z_can = float(sub_conds["z_can"][0].item())

    if max_episode_steps is None and hasattr(env.cfg, "horizon"):
        max_episode_steps = int(env.cfg.horizon)
    if episode_length is None:
        episode_length = int(env.episode_length_buf[0].item())

    is_timeout = max_episode_steps is not None and episode_length >= max_episode_steps - 1

    # --- Classification ---
    # 1. Success
    if in_xy and in_z and lift_latched and released:
        return "success", 1.0

    # 2. Knocked early (during spawn / before engagement)
    if spawn_can_knocked:
        return "knocked_can_early", 0.05

    # 3. Knocked during task
    if can_knocked_now:
        return "knocked_can_during_task", 0.15

    # 4. Never grasped -- hands never came near the can
    if not hands_near_can:
        if is_timeout:
            return "timeout_unengaged", 0.02
        return "never_grasped", 0.10

    # 5. Pick without close -- gripper never closed but can moved up
    if pick_without_close:
        return "grasp_without_lift", 0.20

    # 6. Dropped after lift -- was lifted, then can fell
    if lift_latched and z_can < table_top_z + lift_threshold_m:
        return "dropped_after_lift", 0.25

    # 7. Placement missed -- lift+release ok but not in tray
    if lift_latched and released:
        return "placement_missed", 0.50

    # 8. Timeout with some engagement
    if is_timeout:
        return "timeout_unengaged", 0.08

    return "unknown_failure", 0.05


def compute_a2_quality_score(
    failure_category: str,
    episode_length: int,
    max_episode_steps: int,
    sub_conds: dict[str, torch.Tensor] | None = None,
) -> float:
    """Compute a refined quality score for the episode.

    Args:
        failure_category: The category from classify_a2_failure.
        episode_length: Number of steps taken.
        max_episode_steps: Maximum episode length.
        sub_conds: Optional sub-conditions for finer-grained scoring.

    Returns:
        float: Quality score in [0.0, 1.0].
    """
    if failure_category == "success":
        efficiency = 1.0 - (episode_length / max_episode_steps) * 0.3
        return max(0.70, efficiency)

    base = {
        "knocked_can_early": 0.05,
        "knocked_can_during_task": 0.15,
        "never_grasped": 0.10,
        "grasp_without_lift": 0.20,
        "dropped_after_lift": 0.25,
        "placement_missed": 0.50,
        "timeout_unengaged": 0.02,
        "unknown_failure": 0.05,
    }.get(failure_category, 0.05)

    if sub_conds is not None:
        z_can = float(sub_conds["z_can"][0].item())
        table_z = 1.00
        if abs(z_can - table_z) < 0.03:
            base = min(base + 0.05, 0.70)

    return base
