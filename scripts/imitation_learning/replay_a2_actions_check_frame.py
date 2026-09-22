"""Replay HDF5 actions through Isaac-PickPlace-A2-Abs-v0 to confirm action frame.

Loads demo_0 from the source dataset, resets the env to the recorded initial
state, and feeds raw 38-d actions (Pink-IK schema) into env.step. Each N steps
prints target vs achieved EE pose so we can confirm whether the recorded
``data/demo_*/actions`` is already in env-origin (world) frame — in which case
Pink IK will track within a few cm — or in some other frame, in which case
tracking will diverge identically to the policy eval.

Usage::

    isaaclab.sh -p scripts/imitation_learning/replay_a2_actions_check_frame.py \
        --dataset_file /home/wagner/2T/wagner/dataset/issac_place_can/a2_pickplace_generated.hdf5 \
        --demo demo_0 --max_steps 200 --debug_ik_every 10
"""

from __future__ import annotations

import argparse
import functools
import os
import sys
import time
from pathlib import Path

print = functools.partial(print, flush=True)  # noqa: A001
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import pinocchio  # noqa: F401
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay raw HDF5 actions to verify action frame.")
parser.add_argument("--task_id", type=str, default="Isaac-PickPlace-A2-Abs-v0")
parser.add_argument("--dataset_file", type=str, required=True)
parser.add_argument("--demo", type=str, default="demo_0")
parser.add_argument("--max_steps", type=int, default=200)
parser.add_argument("--debug_ik_every", type=int, default=10)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import h5py
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def _quat_angle_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    q1 = q1 / max(1e-12, float(np.linalg.norm(q1)))
    q2 = q2 / max(1e-12, float(np.linalg.norm(q2)))
    dot = abs(float(np.dot(q1, q2)))
    dot = min(1.0, max(-1.0, dot))
    return float(np.degrees(2.0 * np.arccos(dot)))


def main():
    cfg = parse_env_cfg(args_cli.task_id, device=args_cli.device, num_envs=args_cli.num_envs)
    cfg.teleop_devices.devices = {}
    if hasattr(cfg, "recorders"):
        cfg.recorders = None
    # Show Pink IK warnings if present.
    cfg.actions.upper_body_ik.controller.show_ik_warnings = True
    cfg.actions.upper_body_ik.controller.fail_on_joint_limit_violation = False

    print(f"[replay] making env {args_cli.task_id}")
    env = gym.make(args_cli.task_id, cfg=cfg).unwrapped
    env.seed = args_cli.seed

    body_names = list(env.scene["robot"].data.body_names)
    left_eef_idx = body_names.index("left_arm_link07")
    right_eef_idx = body_names.index("right_arm_link07")
    print(f"[replay] action_dim={env.action_manager.total_action_dim} "
          f"left_eef_idx={left_eef_idx} right_eef_idx={right_eef_idx}")

    print(f"[replay] loading {args_cli.demo} from {args_cli.dataset_file}")
    with h5py.File(args_cli.dataset_file, "r") as f:
        demo = f[f"data/{args_cli.demo}"]
        actions = demo["actions"][:]                    # (T, 38)
        init = demo["initial_state"]
        # robot init
        init_robot_q   = init["articulation/robot/joint_position"][0]   # (53,)
        init_robot_qv  = init["articulation/robot/joint_velocity"][0]
        init_robot_pos = init["articulation/robot/root_pose"][0, :3]
        init_robot_rot = init["articulation/robot/root_pose"][0, 3:7]
        # rigid objects
        init_obj_pose  = init["rigid_object/object/root_pose"][0]
        init_tray_pose = init["rigid_object/tray/root_pose"][0]
    print(f"[replay] {actions.shape[0]} action rows; first row[0:14] (wrist EE poses) =\n  L={actions[0, 0:7]}\n  R={actions[0, 7:14]}")
    print(f"[replay] init root world pose = {init_robot_pos.tolist()}, quat={init_robot_rot.tolist()}")

    # Reset and force initial state to match the demo
    env.reset()
    device = env.device
    env_origin = env.scene.env_origins[0].cpu().numpy()
    print(f"[replay] env_origin = {env_origin.tolist()}")

    robot = env.scene["robot"]
    # Joint state
    q = torch.from_numpy(init_robot_q).to(device, dtype=torch.float32).unsqueeze(0)
    qv = torch.from_numpy(init_robot_qv).to(device, dtype=torch.float32).unsqueeze(0)
    robot.write_joint_state_to_sim(q, qv)
    # Root state (pose + zero velocity)
    root = np.concatenate([init_robot_pos, init_robot_rot, np.zeros(6, dtype=np.float32)]).astype(np.float32)
    robot.write_root_state_to_sim(torch.from_numpy(root).to(device).unsqueeze(0))
    # Objects
    for name, pose in (("object", init_obj_pose), ("tray", init_tray_pose)):
        if name in env.scene.rigid_objects:
            obj = env.scene.rigid_objects[name]
            r = np.concatenate([pose, np.zeros(6, dtype=np.float32)]).astype(np.float32)
            obj.write_root_state_to_sim(torch.from_numpy(r).to(device).unsqueeze(0))

    n = min(args_cli.max_steps, actions.shape[0])
    print(f"[replay] stepping {n} actions through env.step")
    for step in range(n):
        a = actions[step].astype(np.float32)
        action_t = torch.from_numpy(a).to(device).unsqueeze(0)
        env.step(action_t)
        if args_cli.debug_ik_every and step % args_cli.debug_ik_every == 0:
            bs = robot.data.body_state_w[0]
            ach_l = bs[left_eef_idx, :7].detach().cpu().numpy()
            ach_r = bs[right_eef_idx, :7].detach().cpu().numpy()
            ach_l[:3] -= env_origin
            ach_r[:3] -= env_origin
            tgt_l = a[0:7]; tgt_r = a[7:14]
            pos_err_l = float(np.linalg.norm(ach_l[:3] - tgt_l[:3]))
            pos_err_r = float(np.linalg.norm(ach_r[:3] - tgt_r[:3]))
            rot_err_l = _quat_angle_deg(ach_l[3:7], tgt_l[3:7])
            rot_err_r = _quat_angle_deg(ach_r[3:7], tgt_r[3:7])
            print(
                f"  step {step:3d} L tgt_p={tgt_l[:3].round(3).tolist()} "
                f"ach_p={ach_l[:3].round(3).tolist()} |Δp|={pos_err_l:.3f}m angΔ={rot_err_l:5.1f}° | "
                f"R tgt_p={tgt_r[:3].round(3).tolist()} "
                f"ach_p={ach_r[:3].round(3).tolist()} |Δp|={pos_err_r:.3f}m angΔ={rot_err_r:5.1f}°"
            )

    print("[replay] DONE")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
