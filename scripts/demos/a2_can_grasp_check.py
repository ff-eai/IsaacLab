# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kinematic feasibility check: can the A2 s6_hand enclose a standard Nucleus can?

Spawns A2_CFG (5-finger hand) + a pinned can positioned between the palm and
fingertip centroid, drives the hand joints toward their closed limits, and
reports the closest each fingertip gets to the can surface. If any fingertip
fails to approach within a few mm, physical grasp of that can with this hand
geometry is not viable without modifying the URDF joint limits or the grasp
posture.

Usage:
    ./isaaclab.sh -p scripts/demos/a2_can_grasp_check.py           # windowed
    ./isaaclab.sh -p scripts/demos/a2_can_grasp_check.py --headless
    ./isaaclab.sh -p scripts/demos/a2_can_grasp_check.py --side right
    ./isaaclab.sh -p scripts/demos/a2_can_grasp_check.py --can-usd /path/to/other_can.usd
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="A2 5-finger hand × can kinematic feasibility check.")
parser.add_argument("--side", choices=["left", "right"], default="left")
parser.add_argument("--can-usd", default=None, help="Override Nucleus can USD path.")
parser.add_argument("--can-radius", type=float, default=0.0335, help="Assumed can radius in meters (YCB 005 soup: 0.0335).")
parser.add_argument("--steps", type=int, default=240)
parser.add_argument("--contact-threshold-cm", type=float, default=0.5, help="Min fingertip→surface distance to count as 'reached'.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_assets.robots.a2 import A2_CFG

SIDE_PREFIX = "L" if args_cli.side == "left" else "R"
# URDF `left_hand`/`right_hand` is a fixed child of arm_link07 and gets merged away
# by merge_fixed_joints: true in assets/A2/config.yaml. Use arm_link07 as palm.
PALM_LINK = f"{'left' if args_cli.side == 'left' else 'right'}_arm_link07"
FINGERTIP_LINKS = [
    f"{SIDE_PREFIX}_thumb_3",
    f"{SIDE_PREFIX}_index_2",
    f"{SIDE_PREFIX}_middle_2",
    f"{SIDE_PREFIX}_ring_2",
    f"{SIDE_PREFIX}_pinky_2",
]

# Upper joint limits (rad) from A2.urdf, minus 5% margin.
CLOSURE_TARGETS = {
    f"{SIDE_PREFIX}_thumb_swing_joint": 2.17,
    f"{SIDE_PREFIX}_thumb_1_joint": 0.74,
    f"{SIDE_PREFIX}_index_1_joint": 1.61,
    f"{SIDE_PREFIX}_middle_1_joint": 1.61,
    f"{SIDE_PREFIX}_ring_1_joint": 1.61,
    f"{SIDE_PREFIX}_pinky_1_joint": 1.61,
}

DEFAULT_CAN_USD = f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd"


def main() -> None:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 240.0, device=args_cli.device))
    sim.set_camera_view(eye=(0.8, -0.8, 1.6), target=(0.0, 0.0, 1.2))

    sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    robot = Articulation(A2_CFG.replace(prim_path="/World/A2"))

    # Spawn can far away; we'll reposition after we know palm pose.
    can_usd = args_cli.can_usd or DEFAULT_CAN_USD
    can = RigidObject(
        RigidObjectCfg(
            prim_path="/World/Can",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 10.0)),
            spawn=UsdFileCfg(usd_path=can_usd),
        )
    )

    sim.reset()

    # Resolve link and joint indices.
    if PALM_LINK not in robot.body_names:
        raise RuntimeError(f"palm link '{PALM_LINK}' not in body_names: {robot.body_names}")
    missing_tips = [n for n in FINGERTIP_LINKS if n not in robot.body_names]
    if missing_tips:
        raise RuntimeError(f"missing fingertip links: {missing_tips}")

    palm_id = robot.body_names.index(PALM_LINK)
    tip_ids = [robot.body_names.index(n) for n in FINGERTIP_LINKS]

    jnames = robot.joint_names
    name2idx = {n: i for i, n in enumerate(jnames)}

    # Build target joint positions — zero everywhere except the closure targets present on this side.
    joint_pos = robot.data.default_joint_pos.clone()
    driven_targets = {}
    for jname, target in CLOSURE_TARGETS.items():
        if jname in name2idx:
            joint_pos[0, name2idx[jname]] = target
            driven_targets[jname] = target
        else:
            print(f"[a2-can]   WARN: closure joint '{jname}' not found in articulation; skipping")
    print(f"[a2-can] driving {len(driven_targets)} closure joints:")
    for k, v in driven_targets.items():
        print(f"[a2-can]   {k} -> {v:.3f} rad")

    robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))

    # Pin the robot base at its init pose throughout the test.
    pinned_root_pose = robot.data.root_state_w[:, :7].clone()
    pinned_root_vel = torch.zeros_like(robot.data.root_state_w[:, 7:13])

    # Place can midway between palm and fingertip centroid in current pose.
    palm_pos = robot.data.body_pos_w[0, palm_id].detach().cpu().numpy()
    tips_pos = robot.data.body_pos_w[0, tip_ids].detach().cpu().numpy()
    tip_centroid = tips_pos.mean(axis=0)
    can_pos = 0.5 * (palm_pos + tip_centroid)
    print(f"[a2-can] palm pos (world): {palm_pos.round(3).tolist()}")
    print(f"[a2-can] tip centroid:     {tip_centroid.round(3).tolist()}")
    print(f"[a2-can] can target pos:   {can_pos.round(3).tolist()}")

    can_pose = torch.zeros((1, 7), device=sim.device)
    can_pose[0, :3] = torch.tensor(can_pos, device=sim.device)
    can_pose[0, 3] = 1.0  # w component of identity quat
    can.write_root_pose_to_sim(can_pose)
    can.write_root_velocity_to_sim(torch.zeros((1, 6), device=sim.device))

    # Main loop: drive hand closed, keep robot + can pinned, measure gap.
    gap_history = []  # per-step array of 5 distances (tip to can surface, meters)
    for step in range(args_cli.steps):
        robot.set_joint_position_target(joint_pos)
        robot.write_data_to_sim()
        robot.write_root_pose_to_sim(pinned_root_pose)
        robot.write_root_velocity_to_sim(pinned_root_vel)
        can.write_root_pose_to_sim(can_pose)
        can.write_root_velocity_to_sim(torch.zeros((1, 6), device=sim.device))
        sim.step()
        robot.update(sim.get_physics_dt())
        can.update(sim.get_physics_dt())

        cur_can = can.data.root_pos_w[0].detach().cpu().numpy()
        cur_tips = robot.data.body_pos_w[0, tip_ids].detach().cpu().numpy()
        surface_dists = np.linalg.norm(cur_tips - cur_can, axis=-1) - args_cli.can_radius
        gap_history.append(surface_dists)

        if step % 30 == 0 or step == args_cli.steps - 1:
            thumb_tip = cur_tips[0]
            fingers_mean = cur_tips[1:].mean(axis=0)
            t_to_f = np.linalg.norm(thumb_tip - fingers_mean) * 100
            per_tip_str = ", ".join(f"{n.split('_')[1]}={d*100:5.2f}" for n, d in zip(FINGERTIP_LINKS, surface_dists))
            print(f"[a2-can] step={step:3d} | thumb↔fingers-mean={t_to_f:5.2f}cm | tip→surface [cm]: {per_tip_str}")

    gap_arr = np.array(gap_history)  # (steps, 5)
    final_window = gap_arr[-30:]  # last 30 steps after joints settle

    print()
    print("========== VERDICT ==========")
    print(f"Can USD:      {can_usd}")
    print(f"Can radius:   {args_cli.can_radius*100:.2f} cm")
    per_tip_final = final_window.min(axis=0)
    for name, d in zip(FINGERTIP_LINKS, per_tip_final):
        reach = "REACHED" if d * 100 <= args_cli.contact_threshold_cm else "GAP"
        print(f"  {name:<14s}  min dist to can surface = {d*100:6.2f} cm   {reach}")

    all_reached = (per_tip_final * 100 <= args_cli.contact_threshold_cm).all()
    worst = per_tip_final.max() * 100
    thumb_alone = per_tip_final[0] * 100
    fingers_worst = per_tip_final[1:].max() * 100
    print()
    print(f"Worst tip gap: {worst:.2f} cm  (threshold: {args_cli.contact_threshold_cm} cm)")
    print(f"Thumb gap:     {thumb_alone:.2f} cm")
    print(f"Worst finger:  {fingers_worst:.2f} cm")
    if all_reached:
        print("RESULT: full enclosure feasible — proceed with physical grasp pipeline.")
    elif thumb_alone <= args_cli.contact_threshold_cm:
        print("RESULT: thumb reaches but at least one finger does not — power/wrap grasp may still work, precision grasp won't.")
    elif fingers_worst <= args_cli.contact_threshold_cm:
        print("RESULT: fingers reach but thumb does not — confirms RoboTwin gap observation.")
    else:
        print("RESULT: neither side reaches — hand cannot enclose this can with current URDF limits.")


if __name__ == "__main__":
    main()
    simulation_app.close()
