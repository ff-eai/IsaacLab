"""Evaluate LingBot-VLA inside Isaac Lab on Isaac-PickPlace-A2-Abs-v0 — Pink-IK schema.

Companion to ``eval_lingbot_isaac_a2.py``. The other script swapped the action
manager to a 53-joint position term to evaluate the *41-d joint-target* model.
This script keeps the env's native ``PinkInverseKinematicsActionCfg`` and
evaluates the *38-d Pink IK* model:

  state  : 53-d full ``robot_joint_pos`` (env articulation order, matches
           lerobot ``a2_pickplace_v2_39`` ``observation.state.names``)
  action : 38-d  =  left wrist target pose (3+4)
                  + right wrist target pose (3+4)
                  + 24-d hand qpos targets (env ``_HAND_JOINTS`` order)
  IK     : runs inside the env's PinkInverseKinematicsAction term (Pink solves
           arm+waist joints to hit the wrist targets each control tick).

Usage::

    isaaclab.sh -p scripts/imitation_learning/eval_lingbot_isaac_a2_pinkik.py \
        --websocket_host 127.0.0.1 --websocket_port <port> \
        --task "place the can in the tray" --episodes 1 --max_steps 400
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

# Pre-import pinocchio so its pybind11 type casters register before Isaac Sim's
# embedded pybind11 boots (otherwise pinocchio errors on std::vector<string>).
import pinocchio  # noqa: F401

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="LingBot-VLA eval (Pink IK schema).")
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
                    help="If >0, save state + action arrays every N steps.")
parser.add_argument("--debug_dir", type=str, default="/tmp/lingbot_isaac_eval_pinkik")
parser.add_argument("--state_schema", type=str, default="qpos53",
                    choices=["qpos53", "qpos53_ee14"],
                    help="State payload: 53-d qpos (default) or 67-d qpos + 14-d EE pose.")
parser.add_argument("--debug_ik_every", type=int, default=20,
                    help="If >0, print Pink-IK target vs. achieved wrist pose every N steps.")
parser.add_argument("--show_ik_warnings", action="store_true", default=False,
                    help="Enable Pink IK solver warnings (joint-limit hits, low-rank Jacobian, etc.).")
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

import isaaclab_tasks  # noqa: F401  registers task IDs
import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

sys.path.insert(0, args_cli.lerobot_path)
from deploy.websocket_client_policy import WebsocketClientPolicy  # noqa: E402


# Hand-joint mimic children per A2.urdf — slave joints fully determined by their
# parent in the real robot. The IsaacLab no-mimic articulation strips <mimic>
# tags, so the policy can drive these freely; we mask them to the env's default
# (0.0) in both observation and action to match the URDF kinematic constraint.
MIMIC_CHILD_JOINTS = (
    "L_index_2_joint", "L_middle_2_joint", "L_pinky_2_joint", "L_ring_2_joint",
    "R_index_2_joint", "R_middle_2_joint", "R_pinky_2_joint", "R_ring_2_joint",
    "L_thumb_2_joint", "R_thumb_2_joint",
    "L_thumb_3_joint", "R_thumb_3_joint",
)
# Order of the 24-d hand portion of action[14:38] — env's _HAND_JOINTS.
_HAND_JOINT_NAMES_IN_ACTION = (
    "L_index_1_joint", "L_middle_1_joint", "L_pinky_1_joint", "L_ring_1_joint", "L_thumb_swing_joint",
    "R_index_1_joint", "R_middle_1_joint", "R_pinky_1_joint", "R_ring_1_joint", "R_thumb_swing_joint",
    "L_index_2_joint", "L_middle_2_joint", "L_pinky_2_joint", "L_ring_2_joint", "L_thumb_1_joint",
    "R_index_2_joint", "R_middle_2_joint", "R_pinky_2_joint", "R_ring_2_joint", "R_thumb_1_joint",
    "L_thumb_2_joint", "R_thumb_2_joint", "L_thumb_3_joint", "R_thumb_3_joint",
)


def build_mimic_masks(env_joint_names: list[str]):
    """Return (state_indices, action_indices_within_hand) of mimic-child slots."""
    state_idx = [env_joint_names.index(n) for n in MIMIC_CHILD_JOINTS]
    action_idx = [_HAND_JOINT_NAMES_IN_ACTION.index(n) for n in MIMIC_CHILD_JOINTS]
    return np.asarray(state_idx, dtype=np.int64), np.asarray(action_idx, dtype=np.int64)


def encode_obs_for_policy(env, task: str, mimic_state_idx: np.ndarray,
                          state_schema: str,
                          left_eef_idx: int, right_eef_idx: int,
                          env_origin: np.ndarray) -> dict:
    """Build the websocket payload from one Isaac Lab obs dict.

    state_schema:
      ``qpos53``      — 53-d full robot_joint_pos (mimic-child slots zeroed).
      ``qpos53_ee14`` — 67-d: 53-d qpos + left_eef_pos(3) + left_eef_quat(4) +
                       right_eef_pos(3) + right_eef_quat(4), all in env-origin
                       (≈world) frame to match what the converter wrote.
    """
    obs_dict = env.unwrapped.observation_manager.compute()
    pol = obs_dict["policy"]

    def _img_hwc(t: torch.Tensor) -> np.ndarray:
        return t[0].detach().to(torch.uint8).cpu().numpy()

    head = _img_hwc(pol["head_camera_rgb"])
    cleft = _img_hwc(pol["chest_left_camera_rgb"])
    cright = _img_hwc(pol["chest_right_camera_rgb"])
    state_53 = pol["robot_joint_pos"][0].detach().cpu().numpy().astype(np.float32)
    state_53[mimic_state_idx] = 0.0

    if state_schema == "qpos53_ee14":
        bs = env.scene["robot"].data.body_state_w[0]
        l = bs[left_eef_idx, :7].detach().cpu().numpy().astype(np.float32)
        r = bs[right_eef_idx, :7].detach().cpu().numpy().astype(np.float32)
        l[:3] -= env_origin
        r[:3] -= env_origin
        state = np.concatenate([state_53, l[:3], l[3:7], r[:3], r[3:7]])
    else:
        state = state_53

    return {
        "observation.images.cam_high": head,
        "observation.images.cam_left_wrist": cleft,
        "observation.images.cam_right_wrist": cright,
        "observation.state": state,
        "task": task,
    }


def _quat_angle_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    """Geodesic angle (deg) between two unit quaternions in wxyz order."""
    q1 = q1 / max(1e-12, float(np.linalg.norm(q1)))
    q2 = q2 / max(1e-12, float(np.linalg.norm(q2)))
    dot = abs(float(np.dot(q1, q2)))
    dot = min(1.0, max(-1.0, dot))
    return float(np.degrees(2.0 * np.arccos(dot)))


def main():
    cfg = parse_env_cfg(args_cli.task_id, device=args_cli.device, num_envs=args_cli.num_envs)

    # Don't run any teleop devices during eval.
    cfg.teleop_devices.devices = {}
    if hasattr(cfg, "recorders"):
        cfg.recorders = None

    if args_cli.show_ik_warnings:
        cfg.actions.upper_body_ik.controller.show_ik_warnings = True
        cfg.actions.upper_body_ik.controller.fail_on_joint_limit_violation = False

    print(f"[eval] making env {args_cli.task_id} on {args_cli.device}, num_envs={args_cli.num_envs}")
    env = gym.make(args_cli.task_id, cfg=cfg).unwrapped
    env.seed = args_cli.seed

    joint_names = list(env.scene["robot"].data.joint_names)
    print(f"[eval] env articulation has {len(joint_names)} joints")
    print(f"[eval] action term action_dim = {env.action_manager.total_action_dim}")
    if env.action_manager.total_action_dim != 38:
        print(f"  warning: expected 38 (Pink IK schema) but got {env.action_manager.total_action_dim}")
    mimic_state_idx, mimic_hand_idx = build_mimic_masks(joint_names)
    print(f"[eval] forcing mimic-child joints to 0 (state idx={mimic_state_idx.tolist()}, "
          f"action hand idx={mimic_hand_idx.tolist()})")

    # Body indices for Pink IK target frames — used to read achieved EE pose
    # for tracking-error logging. Names taken from pickplace_a2_env_cfg.
    body_names = list(env.scene["robot"].data.body_names)
    try:
        left_eef_idx = body_names.index("left_arm_link07")
        right_eef_idx = body_names.index("right_arm_link07")
    except ValueError as e:
        print(f"[eval] warning: EE link not found in body_names ({e}); IK debug disabled")
        left_eef_idx = right_eef_idx = -1

    print(f"[eval] connecting to ws://{args_cli.websocket_host}:{args_cli.websocket_port}")
    policy = WebsocketClientPolicy(host=args_cli.websocket_host, port=args_cli.websocket_port)
    print(f"[eval] server metadata: {policy.get_server_metadata()}")

    debug_dir = Path(args_cli.debug_dir)
    if args_cli.debug_dump_every:
        debug_dir.mkdir(parents=True, exist_ok=True)

    successes, episode_lengths = 0, []
    for ep in range(args_cli.episodes):
        try:
            policy.reset("isaac_a2")
        except Exception as e:
            print(f"  policy.reset warning: {e}")

        env.reset()
        chunk = None
        success = False
        env_origin = env.scene.env_origins[0].cpu().numpy()
        for step in range(args_cli.max_steps):
            payload = encode_obs_for_policy(
                env, args_cli.task, mimic_state_idx,
                args_cli.state_schema, left_eef_idx, right_eef_idx, env_origin,
            )
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
            if row.shape[0] != 38:
                print(f"  warning: action row dim {row.shape[0]} != 38; using first 38")
                row = row[:38]
            # Force mimic-child action slots to 0 (default joint pose). Hand
            # portion is action[14:38]; mimic_hand_idx is within that 24-d slice.
            row = row.copy()
            row[14 + mimic_hand_idx] = 0.0

            if args_cli.debug_dump_every and step % args_cli.debug_dump_every == 0:
                np.save(debug_dir / f"ep{ep:03d}_step{step:04d}_action.npy", row)

            action_t = torch.from_numpy(row).to(args_cli.device).unsqueeze(0)
            obs, rew, terminated, truncated, info = env.step(action_t)

            # IK tracking debug: target wrist pose came from action; achieved
            # comes from articulation body state in world frame. For num_envs=1
            # env_origin is at world origin so target world = action world.
            if (
                args_cli.debug_ik_every
                and step % args_cli.debug_ik_every == 0
                and left_eef_idx >= 0
            ):
                bs = env.scene["robot"].data.body_state_w[0]
                origin = env.scene.env_origins[0].cpu().numpy()
                ach_l = bs[left_eef_idx, :7].detach().cpu().numpy()
                ach_r = bs[right_eef_idx, :7].detach().cpu().numpy()
                ach_l[:3] -= origin
                ach_r[:3] -= origin
                tgt_l = row[0:7]
                tgt_r = row[7:14]
                pos_err_l = float(np.linalg.norm(ach_l[:3] - tgt_l[:3]))
                pos_err_r = float(np.linalg.norm(ach_r[:3] - tgt_r[:3]))
                rot_err_l = _quat_angle_deg(ach_l[3:7], tgt_l[3:7])
                rot_err_r = _quat_angle_deg(ach_r[3:7], tgt_r[3:7])
                print(
                    f"  step {step:3d} [IK] L tgt_p={tgt_l[:3].round(3).tolist()} "
                    f"ach_p={ach_l[:3].round(3).tolist()} "
                    f"|Δp|={pos_err_l:.3f}m angΔ={rot_err_l:5.1f}° | "
                    f"R tgt_p={tgt_r[:3].round(3).tolist()} "
                    f"ach_p={ach_r[:3].round(3).tolist()} "
                    f"|Δp|={pos_err_r:.3f}m angΔ={rot_err_r:5.1f}°"
                )
            term = bool(terminated[0]) if hasattr(terminated, "__getitem__") else bool(terminated)
            trunc = bool(truncated[0]) if hasattr(truncated, "__getitem__") else bool(truncated)
            if term or trunc:
                success_flag = False
                done_mgr = getattr(env, "termination_manager", None)
                if done_mgr is not None and "success" in done_mgr.active_terms:
                    s_buf = done_mgr.get_term("success")
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
