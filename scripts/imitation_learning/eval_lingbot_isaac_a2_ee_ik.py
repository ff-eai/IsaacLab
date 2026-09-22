"""Closed-loop eval for the lingbot-vla model trained on
``a2_pickplace_v2_combined_ee`` — EE-only arm control via Pink IK.

State:  67-d  (53 env joints + 14 EE pose dims) — same as eval_..._ee_depth.py
Action: model emits 53-d; we drop the arm/waist joint-target slice [14:29]
        and feed only EE targets [0:14] + hand targets [29:53] to the env's
        default Pink-IK ActionsCfg (38-d input: 7+7 EE + 24 hands).
Cameras + depth: 3 RGB + cam_high_depth.

Usage::

    isaaclab.sh -p scripts/imitation_learning/eval_lingbot_isaac_a2_ee_ik.py \\
        --websocket_host 172.18.1.26 --websocket_port 8007 \\
        --task "place the can in the tray" --episodes 1 --max_steps 200 \\
        --use_length 10 --enable_cameras
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

import pinocchio  # noqa: F401  (must come before AppLauncher)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="LingBot-VLA EE+IK eval inside Isaac Lab.")
parser.add_argument("--task_id", type=str, default="Isaac-PickPlace-A2-Abs-v0")
parser.add_argument("--task", type=str, default="place the can in the tray")
parser.add_argument("--websocket_host", type=str, default="172.18.1.26")
parser.add_argument("--websocket_port", type=int, default=8007)
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument("--max_steps", type=int, default=400)
parser.add_argument("--use_length", type=int, default=1)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--lerobot_path", type=str, default="/home/wagner/code/lingbot-vla")
parser.add_argument("--debug_dump_every", type=int, default=0)
parser.add_argument("--debug_dir", type=str, default="/tmp/lingbot_isaac_eval_ee_ik")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.num_envs != 1:
    raise SystemExit("Only num_envs=1 is supported.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

sys.path.insert(0, args_cli.lerobot_path)
from deploy.websocket_client_policy import WebsocketClientPolicy  # noqa: E402


def _img_hwc(t: torch.Tensor) -> np.ndarray:
    return t[0].detach().to(torch.uint8).cpu().numpy()


def _depth_to_3ch_uint8(depth_t: torch.Tensor) -> np.ndarray:
    d = depth_t[0, ..., 0].detach().cpu().numpy().astype(np.float32)
    valid = np.isfinite(d) & (d > 0)
    if valid.any():
        lo = np.percentile(d[valid], 1.0)
        hi = np.percentile(d[valid], 99.0)
    else:
        lo, hi = 0.0, 1.0
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    norm = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
    norm[~valid] = 0.0
    u8 = (norm * 255.0).astype(np.uint8)
    return np.stack([u8, u8, u8], axis=-1)


def encode_obs_for_policy(env, task: str) -> dict:
    pol = env.unwrapped.observation_manager.compute()["policy"]
    head_rgb = _img_hwc(pol["head_camera_rgb"])
    chest_l = _img_hwc(pol["chest_left_camera_rgb"])
    chest_r = _img_hwc(pol["chest_right_camera_rgb"])
    depth_3ch = _depth_to_3ch_uint8(pol["head_camera_depth"])

    joint_pos = pol["robot_joint_pos"][0].detach().cpu().numpy().astype(np.float32)
    le_pos = pol["left_eef_pos"][0].detach().cpu().numpy().astype(np.float32)
    le_quat = pol["left_eef_quat"][0].detach().cpu().numpy().astype(np.float32)
    re_pos = pol["right_eef_pos"][0].detach().cpu().numpy().astype(np.float32)
    re_quat = pol["right_eef_quat"][0].detach().cpu().numpy().astype(np.float32)
    state_67 = np.concatenate([joint_pos, le_pos, le_quat, re_pos, re_quat], axis=0)
    assert state_67.shape == (67,), f"got {state_67.shape}"

    return {
        "observation.images.cam_high": head_rgb,
        "observation.images.cam_left_wrist": chest_l,
        "observation.images.cam_right_wrist": chest_r,
        "observation.images.cam_high_depth": depth_3ch,
        "observation.state": state_67,
        "task": task,
    }


# Hand layout in action[29:53] follows the env's articulation joint order.
# Each pair (index_1/index_2, middle_1/middle_2, ring_1/ring_2, pinky_1/pinky_2,
# thumb_1/thumb_2/thumb_3) is averaged at eval time — same averaged value gets
# replicated to all source positions so the finger acts like a single DOF.
# Index 29 corresponds to hand_tgt_0; indices below are *relative* to
# action[29:53].
#
# 24 finger joint order (matching state[29:53] which we observed in info.json):
#   [0..4]   L_index_1, L_middle_1, L_pinky_1, L_ring_1, L_thumb_swing
#   [5..9]   R_index_1, R_middle_1, R_pinky_1, R_ring_1, R_thumb_swing
#   [10..14] L_index_2, L_middle_2, L_pinky_2, L_ring_2, L_thumb_1
#   [15..19] R_index_2, R_middle_2, R_pinky_2, R_ring_2, R_thumb_1
#   [20]     L_thumb_2
#   [21]     R_thumb_2
#   [22]     L_thumb_3
#   [23]     R_thumb_3
_HAND_GROUPS = [
    [0, 10],          # L_index   = avg(index_1, index_2)
    [1, 11],          # L_middle
    [2, 12],          # L_pinky
    [3, 13],          # L_ring
    [4],              # L_thumb_swing (single)
    [14, 20, 22],     # L_thumb_main = avg(thumb_1, thumb_2, thumb_3)
    [5, 15],          # R_index
    [6, 16],          # R_middle
    [7, 17],          # R_pinky
    [8, 18],          # R_ring
    [9],              # R_thumb_swing
    [19, 21, 23],     # R_thumb_main = avg(thumb_1, thumb_2, thumb_3)
]


def fold_expand_hand_24(hand_24: np.ndarray) -> np.ndarray:
    out = hand_24.astype(np.float32, copy=True)
    for grp in _HAND_GROUPS:
        avg = float(np.mean(out[grp]))
        for k in grp:
            out[k] = avg
    return out


def model_action_to_env_action(action_53: np.ndarray) -> np.ndarray:
    """Drop arm-joint + waist slices; keep EE poses + (folded-then-expanded) hand.

    Layout in:  [L_eef_pos(3) L_eef_quat(4) R_eef_pos(3) R_eef_quat(4)
                 arm_l(7) arm_r(7) waist(1) hand(24)]
    Layout out: [L_eef(7) R_eef(7) hand(24)]  = 38-d
                hand here is the model's 24 outputs after pair-averaging
                each finger so paired _1/_2 joints (and the 3-knuckle thumb)
                track each other, effectively 6 DOF per hand.
    """
    assert action_53.shape == (53,), f"got {action_53.shape}"
    hand = fold_expand_hand_24(action_53[29:53])
    return np.concatenate([action_53[0:14], hand], axis=0)


def main():
    cfg = parse_env_cfg(args_cli.task_id, device=args_cli.device, num_envs=args_cli.num_envs)
    # Keep the env's default Pink-IK ActionsCfg — that's what 'use IK solver
    # to drive arm' means. Just disable teleop/recorders we don't need.
    cfg.teleop_devices.devices = {}
    if hasattr(cfg, "recorders"):
        cfg.recorders = None
    # Raise env episode time-cap so --max_steps controls the horizon. With
    # decimation=6 + physics dt=1/120 → env step = 0.05 s; pad generously.
    cfg.episode_length_s = max(args_cli.max_steps * 0.05 + 5.0, getattr(cfg, "episode_length_s", 0.0))

    print(f"[eval] making env {args_cli.task_id} on {args_cli.device}")
    env = gym.make(args_cli.task_id, cfg=cfg).unwrapped
    env.seed = args_cli.seed

    n_joints = len(list(env.scene["robot"].data.joint_names))
    env_act_dim = env.action_manager.total_action_dim
    print(f"[eval] env articulation has {n_joints} joints, action_dim={env_act_dim}")
    if env_act_dim != 38:
        print(f"  WARN: expected 38-d Pink-IK action input; got {env_act_dim}")

    print(f"[eval] connecting to ws://{args_cli.websocket_host}:{args_cli.websocket_port}")
    policy = WebsocketClientPolicy(host=args_cli.websocket_host, port=args_cli.websocket_port)
    print(f"[eval] server metadata: {policy.get_server_metadata()}")

    debug_dir = Path(args_cli.debug_dir)
    if args_cli.debug_dump_every:
        debug_dir.mkdir(parents=True, exist_ok=True)

    successes, episode_lengths = 0, []
    for ep in range(args_cli.episodes):
        try:
            policy.reset("isaac_a2_ee_ik")
        except Exception as e:
            print(f"  policy.reset warning: {e}")

        env.reset()
        chunk = None
        success = False
        for step in range(args_cli.max_steps):
            payload = encode_obs_for_policy(env, args_cli.task)
            if args_cli.debug_dump_every and step % args_cli.debug_dump_every == 0:
                np.save(debug_dir / f"ep{ep:03d}_step{step:04d}_state.npy",
                        payload["observation.state"])

            if step % max(1, args_cli.use_length) == 0:
                t0 = time.time()
                out = policy.infer(payload)
                action = out.get("action")
                if action is None:
                    print(f"  step {step}: policy returned None — abort")
                    break
                chunk = np.asarray(action, dtype=np.float32)
                if chunk.ndim == 1:
                    chunk = chunk[None, :]
                print(f"  step {step}: chunk shape={chunk.shape} infer={time.time() - t0:.2f}s")

            row = chunk[step % chunk.shape[0]]
            env_row = model_action_to_env_action(row)
            action_t = torch.from_numpy(env_row.astype(np.float32)).to(args_cli.device).unsqueeze(0)
            _, _, terminated, truncated, _ = env.step(action_t)
            term = bool(terminated[0]) if hasattr(terminated, "__getitem__") else bool(terminated)
            trunc = bool(truncated[0]) if hasattr(truncated, "__getitem__") else bool(truncated)
            if term or trunc:
                success_flag = False
                done_mgr = getattr(env, "termination_manager", None)
                if done_mgr is not None:
                    s_buf = done_mgr.get_term("success") if "success" in done_mgr.active_terms else None
                    if s_buf is not None:
                        success_flag = bool(s_buf[0])
                print(f"  step {step}: terminated={term} truncated={trunc} success={success_flag}")
                success = success_flag
                episode_lengths.append(step + 1)
                break
        else:
            episode_lengths.append(args_cli.max_steps)
            print(f"  episode {ep} hit max_steps={args_cli.max_steps}")
        if success:
            successes += 1
        print(f"[eval] episode {ep}: success={success} len={episode_lengths[-1]}")

    print(f"[eval] DONE — {successes}/{args_cli.episodes} success "
          f"(avg len {np.mean(episode_lengths):.1f})")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
