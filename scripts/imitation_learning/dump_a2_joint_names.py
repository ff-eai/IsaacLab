"""Dump A2 mimic env joint name order to JSON.

Run with isaaclab.sh to give the converter a stable name→index mapping
without keeping isaacsim in the conversion's runtime path.
"""

import argparse
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=str, default="/data/home/wagner/code/ext/issac/a2_joint_names.json")
parser.add_argument("--enable_pinocchio", action="store_true", default=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

import isaaclab_mimic.envs  # noqa: F401
import isaaclab_mimic.envs.pinocchio_envs  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main():
    cfg = parse_env_cfg("Isaac-PickPlace-A2-Mimic-v0", device="cpu", num_envs=1)
    cfg.recorders = None
    cfg.terminations = None
    env = gym.make("Isaac-PickPlace-A2-Mimic-v0", cfg=cfg).unwrapped
    names = list(env.scene["robot"].data.joint_names)
    print(f"[dump] {len(names)} joints")
    for i, n in enumerate(names):
        print(f"  [{i:2d}] {n}")
    with open(args_cli.out, "w") as f:
        json.dump({"joint_names": names}, f, indent=2)
    print(f"[dump] wrote {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
