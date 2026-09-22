"""Evaluate LingBot-VLA (raw 53-d state / 38-d Pink-IK action schema) on
Isaac-PickPlace-A2-Mimic-v0.

Differences from the v1 ``eval_lingbot_isaac_a2.py``:
  * Keeps the env's ``PinkInverseKinematicsActionCfg`` action term — does NOT
    swap in a direct 53-joint-position action term. Pink IK absorbs the
    14-d wrist target poses (left + right) and combines them with the 24-d
    hand-joint targets the policy outputs. Total action width = 38.
  * Sends raw 53-d ``obs/robot_joint_pos`` as observation.state.
  * Forwards all four cameras the converter wrote: cam_high (RGB),
    cam_chest_left (RGB), cam_chest_right (RGB), cam_high_depth (depth
    replicated to 3 channels).

Workflow:
  1. Start the lingbot policy WebSocket server with the trained HF checkpoint:
       /home/wagner/code/RoboTwin/.venv/bin/python -m \
           lingbotvla.deploy.lingbot_robotwin_policy \
           --model_path /home/wagner/2T/wagner/trainfromis/checkpoints/global_step_238/hf_ckpt \
           --port 8765 --use_length 50
  2. Run this script:
       isaaclab.sh -p scripts/imitation_learning/eval_lingbot_isaac_a2_raw.py \
           --websocket_host 127.0.0.1 --websocket_port 8765 --episodes 5

Mimic env is used (not the regular Abs env) so the success criterion is the
stateless ``place_on_tray_surface`` (no per-step latches needed during
inference).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task_id", type=str, default="Isaac-PickPlace-A2-Mimic-v0")
parser.add_argument("--task", type=str, default="place the can in the tray")
parser.add_argument("--websocket_host", type=str, default="127.0.0.1")
parser.add_argument("--websocket_port", type=int, default=8765)
parser.add_argument("--episodes", type=int, default=5)
parser.add_argument("--max_steps", type=int, default=400)
parser.add_argument("--use_length", type=int, default=50,
                    help="Number of action-chunk frames to consume between policy.infer calls.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--enable_pinocchio", action="store_true", default=True)
parser.add_argument("--debug_dump_every", type=int, default=0)
parser.add_argument("--debug_dir", type=str, default="/tmp/lingbot_isaac_eval_raw")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = False
args_cli.enable_cameras = True

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# imports below depend on the launched simulation_app
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_mimic.envs  # noqa: F401, E402
import isaaclab_mimic.envs.pinocchio_envs  # noqa: F401, E402
import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

# the lingbot websocket client lives in the lingbot-vla repo; ensure it's importable
LINGBOT_VLA_ROOT = "/home/wagner/code/lingbot-vla"
if LINGBOT_VLA_ROOT not in sys.path:
    sys.path.insert(0, LINGBOT_VLA_ROOT)
from deploy.websocket_client_policy import WebsocketClientPolicy  # noqa: E402


def _img_hwc_uint8(t: torch.Tensor) -> np.ndarray:
    """Convert (B, H, W, 3) uint8 tensor → (H, W, 3) numpy uint8."""
    return t[0].detach().to(torch.uint8).cpu().numpy()


def _depth_hwc_uint8(t: torch.Tensor, max_m: float = 4.0) -> np.ndarray:
    """Convert (B, H, W) or (B, H, W, 1) float32 depth (metres) → (H, W, 3)
    uint8 (replicated channels) clipped to [0, max_m] m and scaled to [0, 255]."""
    a = t[0].detach().cpu().numpy()
    if a.ndim == 3 and a.shape[-1] == 1:
        a = a[..., 0]
    a = np.clip(a, 0.0, max_m)
    u8 = (a / max_m * 255.0).astype(np.uint8)
    return np.stack([u8, u8, u8], axis=-1)


def encode_obs_for_policy(env, task: str) -> dict:
    """Build the websocket payload from one Isaac Lab obs dict — raw schema."""
    obs_dict = env.unwrapped.observation_manager.compute()
    pol = obs_dict["policy"]

    head_rgb = _img_hwc_uint8(pol["head_camera_rgb"])
    chest_l = _img_hwc_uint8(pol["chest_left_camera_rgb"])
    chest_r = _img_hwc_uint8(pol["chest_right_camera_rgb"])
    head_dep = _depth_hwc_uint8(pol["head_camera_depth"])
    state_53 = pol["robot_joint_pos"][0].detach().cpu().numpy().astype(np.float32)

    # The lingbot policy server's resize_image hardcodes the keys
    # `cam_left_wrist` / `cam_right_wrist` (see deploy/lingbot_robotwin_policy.py
    # line 497). Route the chest cameras through those keys — the model only
    # sees tensors, not names.
    return {
        "observation.images.cam_high": head_rgb,
        "observation.images.cam_left_wrist": chest_l,
        "observation.images.cam_right_wrist": chest_r,
        "observation.images.cam_high_depth": head_dep,
        "observation.state": state_53,
        "task": task,
    }


def main():
    cfg = parse_env_cfg(args_cli.task_id, device=args_cli.device, num_envs=args_cli.num_envs)

    # Don't try to run any teleop devices during eval.
    if hasattr(cfg, "teleop_devices") and getattr(cfg.teleop_devices, "devices", None):
        cfg.teleop_devices.devices = {}
    if hasattr(cfg, "recorders"):
        cfg.recorders = None

    print(f"[eval-raw] making env {args_cli.task_id} on {args_cli.device}, num_envs={args_cli.num_envs}")
    env = gym.make(args_cli.task_id, cfg=cfg).unwrapped
    env.seed = args_cli.seed

    print(f"[eval-raw] action term action_dim = {env.action_manager.total_action_dim}")
    print(f"[eval-raw] connecting to ws://{args_cli.websocket_host}:{args_cli.websocket_port}")
    policy = WebsocketClientPolicy(host=args_cli.websocket_host, port=args_cli.websocket_port)
    print(f"[eval-raw] server metadata: {policy.get_server_metadata()}")

    debug_dir = Path(args_cli.debug_dir)
    if args_cli.debug_dump_every:
        debug_dir.mkdir(parents=True, exist_ok=True)

    successes, episode_lengths = 0, []
    for ep in range(args_cli.episodes):
        try:
            policy.reset("isaac_a2_raw")
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
                    print(f"  step {step}: policy returned None — aborting episode")
                    break
                chunk = np.asarray(action, dtype=np.float32)
                if chunk.ndim == 1:
                    chunk = chunk[None, :]
                print(f"  step {step}: chunk shape={chunk.shape} infer={time.time() - t0:.2f}s")
            row = chunk[step % chunk.shape[0]]
            # The model was trained on 38-d actions and is fed through Pink-IK
            # at env-step time. row should already be 38-d; if it's wider (the
            # model padded to max_action_dim=75) take the leading 38.
            action_38 = row[:env.action_manager.total_action_dim]
            action_t = torch.from_numpy(action_38).to(args_cli.device).unsqueeze(0)

            obs, rew, terminated, truncated, info = env.step(action_t)
            term = bool(terminated[0]) if hasattr(terminated, "__getitem__") else bool(terminated)
            trunc = bool(truncated[0]) if hasattr(truncated, "__getitem__") else bool(truncated)
            if term or trunc:
                success_flag = False
                done_mgr = getattr(env, "termination_manager", None)
                if done_mgr is not None:
                    if "success" in done_mgr.active_terms:
                        s_buf = done_mgr.get_term("success")
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
        print(f"[eval-raw] episode {ep}: success={success} len={episode_lengths[-1]}")

    print(f"[eval-raw] DONE — {successes}/{args_cli.episodes} success "
          f"(avg len {np.mean(episode_lengths):.1f})")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
