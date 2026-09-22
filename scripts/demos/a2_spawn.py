# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke test for the A2 humanoid asset.

Two modes:
  --mode usd       : enumerate joints/links directly from the USD (no physics).
                     Use this to diagnose which joints exist and which actuator
                     regex groups miss.
  --mode spawn     : spawn the robot, reset sim, step a few frames, exit.
                     Use this to validate physics loading; may hang if the
                     USD has unresolvable closed-chain joints.

.. code-block:: bash

    ./isaaclab.sh -p scripts/demos/a2_spawn.py --mode usd
    ./isaaclab.sh -p scripts/demos/a2_spawn.py --mode spawn --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="A2 humanoid smoke test.")
parser.add_argument("--mode", choices=["usd", "spawn"], default="usd", help="What to run.")
parser.add_argument("--steps", type=int, default=120, help="Physics steps in spawn mode.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import re

from pxr import Usd, UsdPhysics

from isaaclab_assets.robots.a2 import A2_CFG


def list_joints_from_usd(usd_path: str) -> list[tuple[str, str]]:
    stage = Usd.Stage.Open(usd_path)
    joints: list[tuple[str, str]] = []
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            jtype = prim.GetTypeName()
            joints.append((prim.GetName(), jtype))
    return joints


def run_usd_mode() -> None:
    usd_path = A2_CFG.spawn.usd_path
    print(f"[A2] USD: {usd_path}")
    joints = list_joints_from_usd(usd_path)
    print(f"[A2] total joints in USD: {len(joints)}")
    by_type: dict[str, int] = {}
    for _, t in joints:
        by_type[t] = by_type.get(t, 0) + 1
    for t, c in sorted(by_type.items()):
        print(f"[A2]   {t}: {c}")

    actuator_patterns = {name: cfg.joint_names_expr for name, cfg in A2_CFG.actuators.items()}
    claimed: dict[str, list[str]] = {g: [] for g in actuator_patterns}
    unclaimed: list[str] = []
    for jname, jtype in joints:
        if jtype == "PhysicsFixedJoint":
            continue  # fixed joints don't need actuators
        matched: list[str] = []
        for group, patterns in actuator_patterns.items():
            if any(re.fullmatch(p, jname) for p in patterns):
                matched.append(group)
        if not matched:
            unclaimed.append(jname)
        elif len(matched) > 1:
            print(f"[A2]   CONFLICT: '{jname}' matched by groups {matched}")
        else:
            claimed[matched[0]].append(jname)

    print("[A2] per-group coverage (non-fixed joints):")
    for g, names in claimed.items():
        print(f"[A2]   {g}: {len(names)}")
    if unclaimed:
        print(f"[A2]   UNCLAIMED ({len(unclaimed)}):")
        for n in unclaimed:
            print(f"[A2]     {n}")
    else:
        print("[A2]   UNCLAIMED: none")


def run_spawn_mode() -> None:
    import torch

    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, device=args_cli.device))
    sim.set_camera_view(eye=(3.0, 3.0, 2.0), target=(0.0, 0.0, 1.0))

    sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    print("[A2] constructing Articulation ...")
    robot = Articulation(cfg=A2_CFG.replace(prim_path="/World/A2"))
    print("[A2] calling sim.reset() ...")
    sim.reset()
    print(f"[A2] reset complete — num joints: {robot.num_joints}")
    for n in robot.data.joint_names:
        print(f"[A2]   {n}")
    for group, actuator in robot.actuators.items():
        print(f"[A2]   actuator group '{group}': {len(actuator.joint_names)} joints")

    sim_dt = sim.get_physics_dt()
    zero = torch.zeros_like(robot.data.joint_pos)
    for step in range(args_cli.steps):
        robot.set_joint_position_target(zero)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)
        if step % 30 == 0:
            print(f"[A2] step={step} base_z={robot.data.root_pos_w[0, 2].item():.3f}")
    print("[A2] done.")


if __name__ == "__main__":
    if args_cli.mode == "usd":
        run_usd_mode()
    else:
        run_spawn_mode()
    simulation_app.close()
