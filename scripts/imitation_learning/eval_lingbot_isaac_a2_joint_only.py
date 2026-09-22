"""Closed-loop eval — model output is used as JOINT TARGETS only (no IK).

State:  67-d  (53 env joints + 14 EE pose dims) — same as `..._ee_ik.py`.
Action: build a 53-d JointPositionAction by mapping the model's joint-target
        slice (arms[14:28] + waist[28] + hands[29:53] = 39 values) onto the
        env's articulation joint order, and freezing legs/head at the
        current joint position.

Bypasses Pink IK entirely — the env runs JointPositionActionCfg over all
53 joints.

Usage::

    isaaclab.sh -p scripts/imitation_learning/eval_lingbot_isaac_a2_joint_only.py \\
        --websocket_host 172.18.1.26 --websocket_port 8007 \\
        --task "place the can in the tray" --episodes 1 --max_steps 5000 \\
        --use_length 50 --enable_cameras
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

parser = argparse.ArgumentParser(description="LingBot-VLA joint-only eval inside Isaac Lab.")
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
parser.add_argument("--debug_dir", type=str, default="/tmp/lingbot_isaac_eval_joint_only")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.num_envs != 1:
    raise SystemExit("Only num_envs=1 is supported.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.utils import configclass

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

sys.path.insert(0, args_cli.lerobot_path)
from deploy.websocket_client_policy import WebsocketClientPolicy  # noqa: E402


@configclass
class _JointActOnlyCfg:
    """Single 53-joint position action term — bypasses Pink IK."""

    joint_action: JointPositionActionCfg = JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=1.0,
        use_default_offset=False,
    )


# Model action[14:53] joint-target field names (from the LeRobot meta).
# These map onto specific env joint names.
_MODEL_JOINT_AT = {
    "idx13_left_arm_joint1": 14,
    "idx14_left_arm_joint2": 15,
    "idx15_left_arm_joint3": 16,
    "idx16_left_arm_joint4": 17,
    "idx17_left_arm_joint5": 18,
    "idx18_left_arm_joint6": 19,
    "idx19_left_arm_joint7": 20,
    "idx20_right_arm_joint1": 21,
    "idx21_right_arm_joint2": 22,
    "idx22_right_arm_joint3": 23,
    "idx23_right_arm_joint4": 24,
    "idx24_right_arm_joint5": 25,
    "idx25_right_arm_joint6": 26,
    "idx26_right_arm_joint7": 27,
    "waist_yaw_joint": 28,
    # 24 hand joints — assumes the converter wrote hand_tgt in env articulation
    # joint-name order (training data joint order). The first 5 are L_*_1, then
    # R_*_1, then L_*_2, R_*_2, then L/R_thumb_2 / _3.
    "L_index_1_joint": 29,
    "L_middle_1_joint": 30,
    "L_pinky_1_joint": 31,
    "L_ring_1_joint": 32,
    "L_thumb_swing_joint": 33,
    "R_index_1_joint": 34,
    "R_middle_1_joint": 35,
    "R_pinky_1_joint": 36,
    "R_ring_1_joint": 37,
    "R_thumb_swing_joint": 38,
    "L_index_2_joint": 39,
    "L_middle_2_joint": 40,
    "L_pinky_2_joint": 41,
    "L_ring_2_joint": 42,
    "L_thumb_1_joint": 43,
    "R_index_2_joint": 44,
    "R_middle_2_joint": 45,
    "R_pinky_2_joint": 46,
    "R_ring_2_joint": 47,
    "R_thumb_1_joint": 48,
    "L_thumb_2_joint": 49,
    "R_thumb_2_joint": 50,
    "L_thumb_3_joint": 51,
    "R_thumb_3_joint": 52,
}


def build_index_table(joint_names: list[str]):
    """For each env joint, return (model_action_idx or -1)."""
    out = []
    missing = []
    for name in joint_names:
        idx = _MODEL_JOINT_AT.get(name, -1)
        out.append(idx)
        if idx < 0:
            missing.append(name)
    print(f"[eval] {sum(1 for x in out if x>=0)}/{len(out)} env joints driven from model; "
          f"frozen: {missing}")
    return np.asarray(out, dtype=np.int64)


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


def main():
    cfg = parse_env_cfg(args_cli.task_id, device=args_cli.device, num_envs=args_cli.num_envs)
    cfg.actions = _JointActOnlyCfg()
    cfg.teleop_devices.devices = {}
    if hasattr(cfg, "recorders"):
        cfg.recorders = None
    cfg.episode_length_s = max(args_cli.max_steps * 0.05 + 5.0, getattr(cfg, "episode_length_s", 0.0))

    print(f"[eval] making env {args_cli.task_id} on {args_cli.device}")
    env = gym.make(args_cli.task_id, cfg=cfg).unwrapped
    env.seed = args_cli.seed

    joint_names = list(env.scene["robot"].data.joint_names)
    n_joints = len(joint_names)
    print(f"[eval] env articulation has {n_joints} joints, action_dim={env.action_manager.total_action_dim}")
    map_table = build_index_table(joint_names)

    print(f"[eval] connecting to ws://{args_cli.websocket_host}:{args_cli.websocket_port}")
    policy = WebsocketClientPolicy(host=args_cli.websocket_host, port=args_cli.websocket_port)
    print(f"[eval] server metadata: {policy.get_server_metadata()}")

    debug_dir = Path(args_cli.debug_dir)
    if args_cli.debug_dump_every:
        debug_dir.mkdir(parents=True, exist_ok=True)

    successes, episode_lengths = 0, []
    for ep in range(args_cli.episodes):
        try:
            policy.reset("isaac_a2_joint_only")
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
            assert row.shape[0] == 53, f"unexpected model dim {row.shape}"
            cur = env.scene["robot"].data.joint_pos[0].detach().cpu().numpy().astype(np.float32)
            target = cur.copy()
            mask = map_table >= 0
            target[mask] = row[map_table[mask]]
            action_t = torch.from_numpy(target).to(args_cli.device).unsqueeze(0)
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
