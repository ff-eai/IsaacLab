"""Evaluate LingBot-VLA inside Isaac Lab on Isaac-PickPlace-A2-Abs-v0.

Bypasses the Pink-IK action term and drives the robot directly with
joint-position targets, matching the way training data was reconstructed by
``convert_isaac_a2_to_lerobot.py`` (action[t] = state[t+1] in 41-DOF).

Talks to the LingBot WebSocket policy server (``deploy.lingbot_robotwin_policy``)
running in the lingbot docker container.

Usage::

    isaaclab.sh -p scripts/imitation_learning/eval_lingbot_isaac_a2.py \
        --websocket_host 127.0.0.1 --websocket_port 8765 \
        --task "place the can in the tray" --episodes 1 --max_steps 400
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time
from pathlib import Path

# Ensure prints land in the log even when Isaac Sim shuts down abruptly.
print = functools.partial(print, flush=True)  # noqa: A001
os.environ.setdefault("PYTHONUNBUFFERED", "1")

# Pre-import pinocchio so its pybind11 type casters are registered into the
# correct interpreter before AppLauncher boots Isaac Sim's embedded pybind11.
import pinocchio  # noqa: F401

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="LingBot-VLA eval inside Isaac Lab.")
parser.add_argument("--task_id", type=str, default="Isaac-PickPlace-A2-Abs-v0")
parser.add_argument("--task", type=str, default="place the can in the tray")
parser.add_argument("--websocket_host", type=str, default="127.0.0.1")
parser.add_argument("--websocket_port", type=int, default=8765)
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument("--max_steps", type=int, default=400)
parser.add_argument("--use_length", type=int, default=50,
                    help="Inference cadence: re-query the policy every N env steps.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--lerobot_path", type=str,
                    default="/home/wagner/code/lingbot-vla",
                    help="Repo root containing deploy/websocket_client_policy.py")
parser.add_argument("--debug_dump_every", type=int, default=0,
                    help="If >0, save head/chest images + state CSV every N steps.")
parser.add_argument("--debug_dir", type=str, default="/tmp/lingbot_isaac_eval")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.num_envs != 1:
    raise SystemExit("Only num_envs=1 is supported (one inference stream per env).")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Isaac Lab imports must come after AppLauncher ---------------------------
import gymnasium as gym
import numpy as np
import torch

import isaaclab.envs.mdp as base_mdp
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks  # noqa: F401  registers task IDs
import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

# WebSocket client lives in the lingbot-vla repo; import as a package so its
# relative .msgpack_numpy import resolves.
sys.path.insert(0, args_cli.lerobot_path)
from deploy.websocket_client_policy import WebsocketClientPolicy  # noqa: E402


# 41-DOF lerobot output order (matches convert_isaac_a2_to_lerobot.py).
MOTOR_NAMES = [
    "left_thumb_swing", "left_thumb_main", "left_index", "left_middle", "left_ring", "left_pinky",
    "right_thumb_swing", "right_thumb_main", "right_index", "right_middle", "right_ring", "right_pinky",
    "left_arm_joint1", "left_arm_joint2", "left_arm_joint3",
    "left_arm_joint4", "left_arm_joint5", "left_arm_joint6", "left_arm_joint7",
    "right_arm_joint1", "right_arm_joint2", "right_arm_joint3",
    "right_arm_joint4", "right_arm_joint5", "right_arm_joint6", "right_arm_joint7",
    "waist_yaw",
    "left_hip_roll", "left_hip_yaw", "left_hip_pitch",
    "left_tarsus", "left_toe_pitch", "left_toe_roll",
    "right_hip_roll", "right_hip_yaw", "right_hip_pitch",
    "right_tarsus", "right_toe_pitch", "right_toe_roll",
    "head_joint1", "head_joint2",
]
assert len(MOTOR_NAMES) == 41
MOTOR_INDEX = {n: i for i, n in enumerate(MOTOR_NAMES)}

# (lerobot output joint name) → (env joint name) for the 26 1:1 mappings.
DIRECT_MAP = {
    "left_arm_joint1":  "idx13_left_arm_joint1",
    "left_arm_joint2":  "idx14_left_arm_joint2",
    "left_arm_joint3":  "idx15_left_arm_joint3",
    "left_arm_joint4":  "idx16_left_arm_joint4",
    "left_arm_joint5":  "idx17_left_arm_joint5",
    "left_arm_joint6":  "idx18_left_arm_joint6",
    "left_arm_joint7":  "idx19_left_arm_joint7",
    "right_arm_joint1": "idx20_right_arm_joint1",
    "right_arm_joint2": "idx21_right_arm_joint2",
    "right_arm_joint3": "idx22_right_arm_joint3",
    "right_arm_joint4": "idx23_right_arm_joint4",
    "right_arm_joint5": "idx24_right_arm_joint5",
    "right_arm_joint6": "idx25_right_arm_joint6",
    "right_arm_joint7": "idx26_right_arm_joint7",
    "waist_yaw":        "waist_yaw_joint",
    "left_hip_roll":    "idx01_left_hip_roll",
    "left_hip_yaw":     "idx02_left_hip_yaw",
    "left_hip_pitch":   "idx03_left_hip_pitch",
    "left_tarsus":      "idx04_left_tarsus",
    "left_toe_pitch":   "idx05_left_toe_pitch",
    "left_toe_roll":    "idx06_left_toe_roll",
    "right_hip_roll":   "idx07_right_hip_roll",
    "right_hip_yaw":    "idx08_right_hip_yaw",
    "right_hip_pitch":  "idx09_right_hip_pitch",
    "right_tarsus":     "idx10_right_tarsus",
    "right_toe_pitch":  "idx11_right_toe_pitch",
    "right_toe_roll":   "idx12_right_toe_roll",
    "head_joint1":      "idx27_head_joint1",
    "head_joint2":      "idx28_head_joint2",
}

# (lerobot finger output name) → list of (env joint name, expansion ratio).
# Recording averaged the same-finger group; at eval we invert that average
# under the URDF mimic relations stripped during USD conversion:
#   finger_2 = 1.0 × finger_1                (index/middle/ring/pinky)
#   thumb_2  = 0.40 × thumb_1, thumb_3 = 0.60 × thumb_1
# avg = (1 + r2 + r3) / 3 × parent
# parent = avg / parent_factor; child_k = ratio_k × parent.
HAND_GROUPS = {
    "left_thumb_swing":  [("L_thumb_swing_joint", 1.0)],
    "right_thumb_swing": [("R_thumb_swing_joint", 1.0)],
    "left_thumb_main":   [
        ("L_thumb_1_joint", 1.0 / ((1.0 + 0.40 + 0.60) / 3.0)),
        ("L_thumb_2_joint", 0.40 / ((1.0 + 0.40 + 0.60) / 3.0)),
        ("L_thumb_3_joint", 0.60 / ((1.0 + 0.40 + 0.60) / 3.0)),
    ],
    "right_thumb_main":  [
        ("R_thumb_1_joint", 1.0 / ((1.0 + 0.40 + 0.60) / 3.0)),
        ("R_thumb_2_joint", 0.40 / ((1.0 + 0.40 + 0.60) / 3.0)),
        ("R_thumb_3_joint", 0.60 / ((1.0 + 0.40 + 0.60) / 3.0)),
    ],
    "left_index":  [("L_index_1_joint", 1.0),  ("L_index_2_joint", 1.0)],
    "left_middle": [("L_middle_1_joint", 1.0), ("L_middle_2_joint", 1.0)],
    "left_ring":   [("L_ring_1_joint", 1.0),   ("L_ring_2_joint", 1.0)],
    "left_pinky":  [("L_pinky_1_joint", 1.0),  ("L_pinky_2_joint", 1.0)],
    "right_index":  [("R_index_1_joint", 1.0),  ("R_index_2_joint", 1.0)],
    "right_middle": [("R_middle_1_joint", 1.0), ("R_middle_2_joint", 1.0)],
    "right_ring":   [("R_ring_1_joint", 1.0),   ("R_ring_2_joint", 1.0)],
    "right_pinky":  [("R_pinky_1_joint", 1.0),  ("R_pinky_2_joint", 1.0)],
}


@configclass
class _JointActOnlyCfg:
    """Replacement for ``ActionsCfg`` — single 53-joint position action term."""

    joint_action: JointPositionActionCfg = JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],            # all joints
        scale=1.0,
        offset=0.0,
        use_default_offset=False,
        preserve_order=False,
    )


def build_state_projection(env_joint_names: list[str]):
    """Forward (53-d → 41-d) projection used to encode observation.state."""
    name_to_env = {n: i for i, n in enumerate(env_joint_names)}
    direct_pairs = []
    for out_name, src_name in DIRECT_MAP.items():
        direct_pairs.append((MOTOR_INDEX[out_name], name_to_env[src_name]))
    avg_groups = []
    for out_name, srcs in HAND_GROUPS.items():
        env_idxs = [name_to_env[joint] for joint, _ratio in srcs]
        avg_groups.append((MOTOR_INDEX[out_name], env_idxs))
    return np.asarray(direct_pairs, dtype=np.int64), avg_groups


def project_state_53_to_41(joint_pos_53: np.ndarray, direct_pairs, avg_groups) -> np.ndarray:
    out = np.zeros((41,), dtype=np.float32)
    out[direct_pairs[:, 0]] = joint_pos_53[direct_pairs[:, 1]]
    for out_idx, env_idxs in avg_groups:
        out[out_idx] = float(np.mean([joint_pos_53[i] for i in env_idxs]))
    return out


def build_action_expansion(env_joint_names: list[str]):
    """Inverse (41-d → 53-d) expansion used to drive joint targets each step."""
    name_to_env = {n: i for i, n in enumerate(env_joint_names)}
    pairs = []  # (env_joint_idx, motor_idx, ratio)
    for motor_name, env_name in DIRECT_MAP.items():
        pairs.append((name_to_env[env_name], MOTOR_INDEX[motor_name], 1.0))
    for motor_name, srcs in HAND_GROUPS.items():
        for env_name, ratio in srcs:
            pairs.append((name_to_env[env_name], MOTOR_INDEX[motor_name], ratio))
    pairs.sort()
    env_idx = np.asarray([p[0] for p in pairs], dtype=np.int64)
    motor_idx = np.asarray([p[1] for p in pairs], dtype=np.int64)
    ratio = np.asarray([p[2] for p in pairs], dtype=np.float32)
    return env_idx, motor_idx, ratio


def expand_action_41_to_53(action_41: np.ndarray, env_idx, motor_idx, ratio,
                           default_pose: np.ndarray) -> np.ndarray:
    out = default_pose.copy()
    out[env_idx] = action_41[motor_idx] * ratio
    return out


def encode_obs_for_policy(env, task: str, direct_pairs, avg_groups) -> dict:
    """Build the websocket payload from one Isaac Lab obs dict."""
    obs_dict = env.unwrapped.observation_manager.compute()
    pol = obs_dict["policy"]

    def _img_hwc(t: torch.Tensor) -> np.ndarray:
        # mdp.image with normalize=False returns (B, H, W, 3) uint8.
        a = t[0].detach().to(torch.uint8).cpu().numpy()
        return a

    head = _img_hwc(pol["head_camera_rgb"])
    cleft = _img_hwc(pol["chest_left_camera_rgb"])
    cright = _img_hwc(pol["chest_right_camera_rgb"])
    joint_pos_53 = pol["robot_joint_pos"][0].detach().cpu().numpy().astype(np.float32)
    state_41 = project_state_53_to_41(joint_pos_53, direct_pairs, avg_groups)

    return {
        # The lingbot inference Normalizer keys on cam_left_wrist / cam_right_wrist;
        # we route chest_left / chest_right images through those keys (the model
        # only sees tensors, not names).
        "observation.images.cam_high": head,
        "observation.images.cam_left_wrist": cleft,
        "observation.images.cam_right_wrist": cright,
        "observation.state": state_41,
        "task": task,
    }


def main():
    cfg = parse_env_cfg(args_cli.task_id, device=args_cli.device, num_envs=args_cli.num_envs)

    # Replace the Pink-IK action term with a direct 53-joint position action.
    # parse_env_cfg has already run __post_init__, so the URDF path on the IK
    # cfg is set but no IK controller has been instantiated yet — the term
    # gets discarded entirely once we overwrite ``cfg.actions`` here.
    cfg.actions = _JointActOnlyCfg()

    # Don't try to run any teleop devices during eval.
    cfg.teleop_devices.devices = {}

    # We don't need a recorder either.
    if hasattr(cfg, "recorders"):
        cfg.recorders = None

    print(f"[eval] making env {args_cli.task_id} on {args_cli.device}, num_envs={args_cli.num_envs}")
    env = gym.make(args_cli.task_id, cfg=cfg).unwrapped
    env.seed = args_cli.seed

    joint_names = list(env.scene["robot"].data.joint_names)
    print(f"[eval] env articulation has {len(joint_names)} joints")
    direct_pairs, avg_groups = build_state_projection(joint_names)
    env_idx, motor_idx, ratio = build_action_expansion(joint_names)

    default_pose = (
        env.scene["robot"].data.default_joint_pos[0].detach().cpu().numpy().astype(np.float32)
    )
    print(f"[eval] action term action_dim = {env.action_manager.total_action_dim}")
    if env.action_manager.total_action_dim != len(joint_names):
        print(f"  warning: expected {len(joint_names)} but got {env.action_manager.total_action_dim}")

    # WebSocket policy ------------------------------------------------------
    print(f"[eval] connecting to ws://{args_cli.websocket_host}:{args_cli.websocket_port}")
    policy = WebsocketClientPolicy(host=args_cli.websocket_host, port=args_cli.websocket_port)
    print(f"[eval] server metadata: {policy.get_server_metadata()}")

    debug_dir = Path(args_cli.debug_dir)
    if args_cli.debug_dump_every:
        debug_dir.mkdir(parents=True, exist_ok=True)

    successes, episode_lengths = 0, []
    for ep in range(args_cli.episodes):
        # Reset per-episode state on the policy server (clears chunk cache).
        try:
            policy.reset("isaac_a2")
        except Exception as e:
            print(f"  policy.reset warning: {e}")

        env.reset()
        chunk = None
        success = False
        for step in range(args_cli.max_steps):
            payload = encode_obs_for_policy(env, args_cli.task, direct_pairs, avg_groups)

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
            action_53 = expand_action_41_to_53(row, env_idx, motor_idx, ratio, default_pose)
            action_t = torch.from_numpy(action_53).to(args_cli.device).unsqueeze(0)

            obs, rew, terminated, truncated, info = env.step(action_t)
            term = bool(terminated[0]) if hasattr(terminated, "__getitem__") else bool(terminated)
            trunc = bool(truncated[0]) if hasattr(truncated, "__getitem__") else bool(truncated)
            if term or trunc:
                # Inspect terminations.success — exists when DoneTerm 'success' fired.
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
