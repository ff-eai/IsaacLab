"""Closed-loop eval for LingBot-VLA v2 WebSocket policy on A2-OmniHand.

Pairs with ``deploy.lingbot_vla_v2_policy`` (default port 8006). Uses the
``robotwin`` robot config on the server: 14-d interleaved state/action
(left arm 6 + left effector + right arm 6 + right effector). Maps Isaac's
7+7 arm joints into that layout and drives the env with joint-position
targets; non-arm joints (legs, waist, head, hands) stay at their current qpos.

Usage::

    isaaclab.sh -p scripts/imitation_learning/eval_lingbot_v2_isaac_a2omnihand.py \\
        --websocket_host 127.0.0.1 --websocket_port 8006 \\
        --episodes 1 --max_steps 400 --use_length 25 \\
        --headless --enable_cameras --device cuda:0
"""

from __future__ import annotations

import argparse
import functools
import os
import sys
import time

print = functools.partial(print, flush=True)  # noqa: A001
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import pinocchio  # noqa: F401

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="LingBot-VLA v2 eval on A2-OmniHand.")
parser.add_argument("--task_id", type=str, default="Isaac-PickPlace-A2OmniHand-Abs-v0")
parser.add_argument("--task", type=str, default="place the coca-cola bottle in the tray")
parser.add_argument("--websocket_host", type=str, default="127.0.0.1")
parser.add_argument("--websocket_port", type=int, default=8006)
parser.add_argument("--robo_name", type=str, default="robotwin",
                    help="Robot config name passed to policy.reset() on the v2 server.")
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument("--max_steps", type=int, default=400)
parser.add_argument("--use_length", type=int, default=25)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--lerobot_path", type=str, default="/home/wagner/code/lingbot-vla-v2")
parser.add_argument(
    "--bottle_side",
    type=str,
    default="right",
    choices=("left", "right", "random"),
)
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
from isaaclab_tasks.manager_based.manipulation.pick_place.pickplace_a2_omnihand_env_cfg import (
    enable_pickplace_omnihand_5cameras,
)
from isaaclab_tasks.utils import parse_env_cfg

sys.path.insert(0, args_cli.lerobot_path)
from deploy.websocket_client_policy import WebsocketClientPolicy  # noqa: E402

ARM_JOINT_NAMES = [
    "idx13_left_arm_joint1", "idx14_left_arm_joint2", "idx15_left_arm_joint3",
    "idx16_left_arm_joint4", "idx17_left_arm_joint5", "idx18_left_arm_joint6",
    "idx19_left_arm_joint7",
    "idx20_right_arm_joint1", "idx21_right_arm_joint2", "idx22_right_arm_joint3",
    "idx23_right_arm_joint4", "idx24_right_arm_joint5", "idx25_right_arm_joint6",
    "idx26_right_arm_joint7",
]


@configclass
class _JointActOnlyCfg:
    joint_action: JointPositionActionCfg = JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=1.0,
        use_default_offset=False,
    )


def _img_hwc(t: torch.Tensor) -> np.ndarray:
    return t[0].detach().to(torch.uint8).cpu().numpy()


def arm14_to_robotwin_state(arm14: np.ndarray, left_eff: float, right_eff: float) -> np.ndarray:
    """Pack 7+7 arm joints into robotwin's 14-d interleaved layout."""
    out = np.zeros(14, dtype=np.float32)
    out[0:6] = arm14[0:6]
    out[6] = left_eff
    out[7:13] = arm14[7:13]
    out[13] = right_eff
    return out


def robotwin_action_to_arm14(action14: np.ndarray, arm14_now: np.ndarray) -> np.ndarray:
    """Unpack robotwin 14-d action chunk row into 7+7 arm joint targets."""
    out = arm14_now.astype(np.float32, copy=True)
    out[0:6] = action14[0:6]
    out[7:13] = action14[7:13]
    return out


def encode_obs(env, arm14: np.ndarray, left_eff: float, right_eff: float, task: str) -> dict:
    pol = env.unwrapped.observation_manager.compute()["policy"]
    return {
        "observation.images.cam_high": _img_hwc(pol["head_camera_rgb"]),
        "observation.images.cam_left_wrist": _img_hwc(pol["left_wrist_camera_rgb"]),
        "observation.images.cam_right_wrist": _img_hwc(pol["right_wrist_camera_rgb"]),
        "observation.state": arm14_to_robotwin_state(arm14, left_eff, right_eff),
        "task": task,
    }


def hand_effector_proxy(joint_pos: np.ndarray, hand_idx: np.ndarray) -> float:
    """Scalar gripper proxy for robotwin effector slots (mean active hand qpos)."""
    if hand_idx is None or len(hand_idx) == 0:
        return 0.0
    return float(np.mean(joint_pos[hand_idx]))


def main():
    cfg = parse_env_cfg(args_cli.task_id, device=args_cli.device, num_envs=args_cli.num_envs)
    enable_pickplace_omnihand_5cameras(cfg)
    cfg.actions = _JointActOnlyCfg()
    cfg.teleop_devices.devices = {}
    if hasattr(cfg, "recorders"):
        cfg.recorders = None
    cfg.episode_length_s = max(args_cli.max_steps * 0.05 + 5.0, getattr(cfg, "episode_length_s", 0.0))
    if args_cli.bottle_side != "random":
        cfg.events.reset_object.params["force_side"] = args_cli.bottle_side

    print(f"[eval] env={args_cli.task_id} cameras=head+chest+wrist ws://{args_cli.websocket_host}:{args_cli.websocket_port}")
    env = gym.make(args_cli.task_id, cfg=cfg).unwrapped
    env.seed = args_cli.seed

    env_joint_names = list(env.scene["robot"].data.joint_names)
    name_to_idx = {n: i for i, n in enumerate(env_joint_names)}
    arm_env_idx = np.asarray([name_to_idx[n] for n in ARM_JOINT_NAMES], dtype=np.int64)

    hand_active_idx = None
    for names in (
        [
            "L_thumb_roll_joint", "L_thumb_abad_joint", "L_thumb_mcp_joint",
            "L_index_abad_joint", "L_index_pip_joint", "L_middle_pip_joint",
            "L_ring_abad_joint", "L_ring_pip_joint", "L_pinky_abad_joint", "L_pinky_pip_joint",
            "R_thumb_roll_joint", "R_thumb_abad_joint", "R_thumb_mcp_joint",
            "R_index_abad_joint", "R_index_pip_joint", "R_middle_pip_joint",
            "R_ring_abad_joint", "R_ring_pip_joint", "R_pinky_abad_joint", "R_pinky_pip_joint",
        ],
    ):
        if all(n in name_to_idx for n in names):
            hand_active_idx = np.asarray([name_to_idx[n] for n in names], dtype=np.int64)
            break

    left_hand_idx = hand_active_idx[:10] if hand_active_idx is not None else None
    right_hand_idx = hand_active_idx[10:20] if hand_active_idx is not None else None

    policy = WebsocketClientPolicy(
        host=args_cli.websocket_host, port=args_cli.websocket_port,
    )
    print(f"[eval] server metadata: {policy.get_server_metadata()}")

    successes, episode_lengths = 0, []
    for ep in range(args_cli.episodes):
        try:
            policy.reset(args_cli.robo_name)
        except Exception as e:
            print(f"  policy.reset warning: {e}")

        env.reset()
        chunk = None
        success = False
        for step in range(args_cli.max_steps):
            cur = env.scene["robot"].data.joint_pos[0].detach().cpu().numpy().astype(np.float32)
            arm14 = cur[arm_env_idx]
            left_eff = hand_effector_proxy(cur, left_hand_idx)
            right_eff = hand_effector_proxy(cur, right_hand_idx)
            payload = encode_obs(env, arm14, left_eff, right_eff, args_cli.task)

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
            if row.shape[0] != 14:
                print(f"  warning: expected 14-d action row, got {row.shape[0]}")
                row = row[:14]

            target = cur.copy()
            target[arm_env_idx] = robotwin_action_to_arm14(row, arm14)
            action_t = torch.from_numpy(target).to(args_cli.device).unsqueeze(0)
            _, _, terminated, truncated, _ = env.step(action_t)

            term = bool(terminated[0]) if hasattr(terminated, "__getitem__") else bool(terminated)
            trunc = bool(truncated[0]) if hasattr(truncated, "__getitem__") else bool(truncated)
            if term or trunc:
                success_flag = False
                done_mgr = getattr(env, "termination_manager", None)
                if done_mgr is not None and "success" in done_mgr.active_terms:
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
        print(f"[eval] episode {ep}: success={success} len={episode_lengths[-1]}")

    print(f"[eval] DONE — {successes}/{args_cli.episodes} success "
          f"(avg len {np.mean(episode_lengths):.1f})")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
