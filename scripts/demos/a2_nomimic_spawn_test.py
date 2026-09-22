"""Spawn + physics smoke test for A2_NOMIMIC_CFG.

Purpose: verify the nomimic USD no longer hangs PhysX. Spawns robot, steps 120
frames, exits. If it exits cleanly we can build the task env on top of it.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=120)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app = AppLauncher(args).app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

from isaaclab_assets.robots.a2 import A2_NOMIMIC_CFG


def main() -> None:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, device=args.device))
    sim.set_camera_view(eye=(3.0, 3.0, 2.0), target=(0.0, 0.0, 1.0))
    sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2000.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=2000.0))

    print("[nomimic] constructing Articulation ...")
    robot = Articulation(A2_NOMIMIC_CFG.replace(prim_path="/World/A2"))
    print("[nomimic] calling sim.reset() ...")
    sim.reset()
    print(f"[nomimic] reset OK — num_joints={robot.num_joints}, num_bodies={len(robot.body_names)}")

    hand_joints = [n for n in robot.joint_names if n.startswith("L_") or n.startswith("R_")]
    print(f"[nomimic] hand joints found: {len(hand_joints)}")
    for n in hand_joints:
        print(f"[nomimic]   {n}")

    sim_dt = sim.get_physics_dt()
    zero = torch.zeros_like(robot.data.joint_pos)
    for step in range(args.steps):
        robot.set_joint_position_target(zero)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)
        if step % 30 == 0 or step == args.steps - 1:
            print(f"[nomimic] step={step} base_z={robot.data.root_pos_w[0, 2].item():.3f}")
    print("[nomimic] SUCCESS — physics stepped without hang.")


if __name__ == "__main__":
    main()
    app.close()
