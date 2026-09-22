"""Sweep A2's wrist joints (5/6/7) through their full URDF ranges to verify
the kinematic behavior independent of Pink IK.

Spawns the A2 robot, holds all non-wrist joints at zero, and cycles each wrist
joint individually through its [lower, upper] limit so you can visually confirm
which joint produces which physical motion (forearm twist, wrist flexion,
wrist deviation).

Run:
    ./isaaclab.sh -p scripts/demos/a2_wrist_sweep.py
    # add --headless to skip the GUI
    # add --joint joint5 to sweep only one joint
"""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--joint", choices=["joint5", "joint6", "joint7", "all"], default="all")
parser.add_argument("--side", choices=["left", "right", "both"], default="both")
parser.add_argument("--cycle_seconds", type=float, default=4.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app = AppLauncher(args).app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab_assets.robots.a2 import A2_NOMIMIC_CFG  # isort: skip

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, device=args.device))
sim.set_camera_view([1.5, 1.5, 1.4], [0.0, 0.0, 1.1])

sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())
sim_utils.DomeLightCfg(intensity=2000.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=2000.0))

robot_cfg = A2_NOMIMIC_CFG.replace(prim_path="/World/A2")
robot_cfg.spawn.func(robot_cfg.prim_path, robot_cfg.spawn, translation=(0.0, 0.0, 1.05))
robot = Articulation(robot_cfg)
sim.reset()

# Per-side wrist joints: (joint5, joint6, joint7)
WRIST_JOINTS = {
    "left":  ["idx17_left_arm_joint5",  "idx18_left_arm_joint6",  "idx19_left_arm_joint7"],
    "right": ["idx24_right_arm_joint5", "idx25_right_arm_joint6", "idx26_right_arm_joint7"],
}

# Resolve indices in the asset's joint ordering.
joint_ids: dict[str, dict[str, int]] = {"left": {}, "right": {}}
for side, names in WRIST_JOINTS.items():
    ids, found_names = robot.find_joints(names)
    for jname, jid in zip(found_names, ids):
        # Map back to "joint5" / "joint6" / "joint7" key.
        for short in ("joint5", "joint6", "joint7"):
            if short in jname:
                joint_ids[side][short] = jid

print(f"[sweep] resolved joint ids: {joint_ids}")

# Hold all joints at zero except the one we're sweeping.
zero_pose = torch.zeros((1, robot.num_joints), device=args.device)
robot.write_joint_state_to_sim(zero_pose, torch.zeros_like(zero_pose))

# Joint URDF limits (from A2.urdf).
LIMITS = {
    "joint5": (-2.879, 2.879),
    "joint6": (-0.785, 0.785),
    "joint7": (-1.0472, 1.0472),  # widened version
}

# Sweep loop.
sides_to_sweep = ["left", "right"] if args.side == "both" else [args.side]
joints_to_sweep = ["joint5", "joint6", "joint7"] if args.joint == "all" else [args.joint]

steps_per_cycle = int(args.cycle_seconds / sim.get_physics_dt())
phase = 0.0
phase_step = 2 * math.pi / steps_per_cycle
joint_idx_in_cycle = 0
cycle_count = 0

print(f"[sweep] sweeping joints {joints_to_sweep} on sides {sides_to_sweep} "
      f"({args.cycle_seconds}s per joint per side, {steps_per_cycle} steps each)")

while app.is_running():
    # Pick the current sweep target by cycle.
    cycle_index = (cycle_count // (len(sides_to_sweep) * len(joints_to_sweep))) % 1
    pair_idx = cycle_count % (len(sides_to_sweep) * len(joints_to_sweep))
    side = sides_to_sweep[pair_idx // len(joints_to_sweep)]
    joint_short = joints_to_sweep[pair_idx % len(joints_to_sweep)]
    jid = joint_ids[side][joint_short]
    lo, hi = LIMITS[joint_short]
    mid = 0.5 * (lo + hi)
    amp = 0.5 * (hi - lo)
    target_val = mid + amp * math.sin(phase)

    pose = zero_pose.clone()
    pose[0, jid] = target_val
    robot.write_joint_state_to_sim(pose, torch.zeros_like(pose))
    robot.set_joint_position_target(pose)
    robot.write_data_to_sim()

    sim.step()
    robot.update(sim.get_physics_dt())

    phase += phase_step
    if phase >= 2 * math.pi:
        phase -= 2 * math.pi
        cycle_count += 1
        next_pair_idx = cycle_count % (len(sides_to_sweep) * len(joints_to_sweep))
        next_side = sides_to_sweep[next_pair_idx // len(joints_to_sweep)]
        next_joint = joints_to_sweep[next_pair_idx % len(joints_to_sweep)]
        print(f"[sweep] now sweeping {next_side} {next_joint}")

app.close()
