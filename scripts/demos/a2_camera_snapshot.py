"""Dump head_camera/chest_left/chest_right frames + head pose to PNGs.

One-shot diagnostic: spawn the OmniHand env, reset once, save the RGB obs
of each camera as PNG, print the head_link02 world pose. Use to verify what
the policy sees and to compute a "look-at-table" rotation.

Usage:
    python scripts/demos/a2_camera_snapshot.py [--task_id Isaac-PickPlace-A2OmniHand-Abs-v0]
"""
from __future__ import annotations

import argparse
import os
import sys

import pinocchio  # noqa: F401  (must come before AppLauncher)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task_id", default="Isaac-PickPlace-A2OmniHand-Abs-v0")
parser.add_argument("--out_dir", default="/tmp/camera_snapshot")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from PIL import Image

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main():
    os.makedirs(args_cli.out_dir, exist_ok=True)
    cfg = parse_env_cfg(args_cli.task_id, device=args_cli.device, num_envs=1)
    cfg.teleop_devices.devices = {}
    if hasattr(cfg, "recorders"):
        cfg.recorders = None
    env = gym.make(args_cli.task_id, cfg=cfg).unwrapped

    # Step a few times to let physics settle / cameras render.
    env.reset()
    for _ in range(5):
        zero_action = torch.zeros(1, env.action_manager.total_action_dim, device=args_cli.device)
        env.step(zero_action)

    pol = env.observation_manager.compute()["policy"]

    for body in ("base_link", "head_link02", "head_link01", "head_link03",
                 "left_arm_link07", "right_arm_link07"):
        try:
            idx = env.scene["robot"].data.body_names.index(body)
            pose = env.scene["robot"].data.body_state_w[0, idx, :7].cpu().numpy()
            print(f"[snap] {body:24s} pos={pose[:3].round(3).tolist()} quat_wxyz={pose[3:].round(3).tolist()}")
        except (ValueError, Exception) as e:
            print(f"[snap] {body}: skip ({e})")

    # Save camera RGB obs.
    for key in ("head_camera_rgb", "chest_left_camera_rgb", "chest_right_camera_rgb"):
        try:
            img = pol[key][0].detach().to(torch.uint8).cpu().numpy()
            path = os.path.join(args_cli.out_dir, f"{key}.png")
            Image.fromarray(img).save(path)
            print(f"[snap] saved {path}  shape={img.shape}")
        except Exception as e:
            print(f"[snap] {key}: skip ({e})")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
