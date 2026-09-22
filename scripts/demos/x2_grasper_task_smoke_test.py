"""Smoke test for Isaac-PickPlace-X2-Grasper-Abs-v0."""

from __future__ import annotations

import argparse
import sys
import traceback

import pinocchio as _pin_preload  # noqa: F401
from pinocchio.robot_wrapper import RobotWrapper as _RW_preload  # noqa: F401

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import torch

from isaaclab_tasks.manager_based.manipulation.pick_place.pickplace_x2_grasper_env_cfg import (
    PickPlaceX2GrasperEnvCfg,
)


def main() -> int:
    task_id = "Isaac-PickPlace-X2-Grasper-Abs-v0"
    print(f"[smoke] checking registry for {task_id}")
    if task_id not in gym.envs.registry:
        print(f"[smoke] FAIL: {task_id} not registered")
        return 1

    cfg = PickPlaceX2GrasperEnvCfg()
    print(f"[smoke] idle_action dim={len(cfg.idle_action)}")
    print(f"[smoke] urdf={cfg.actions.upper_body_ik.controller.urdf_path}")

    print(f"[smoke] gym.make({task_id})")
    env = gym.make(task_id, cfg=cfg)
    print(f"[smoke] action_space={env.action_space}")

    obs, info = env.reset()
    print(f"[smoke] reset OK")

    idle = cfg.idle_action.unsqueeze(0).to(env.unwrapped.device)
    obs, rew, term, trunc, info = env.step(idle)
    print(f"[smoke] step OK term={term.any().item()} trunc={trunc.any().item()}")

    env.close()
    app.close()
    print("[smoke] all passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        app.close()
        raise SystemExit(1) from None
