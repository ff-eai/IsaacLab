"""Closed-loop EE+IK eval with adaptive chunk reuse.

Same model/observation/action wiring as ``eval_lingbot_isaac_a2_ee_ik.py``,
but the policy-query cadence (``use_length``) switches based on a 3-phase
state machine:

  approach   eef-to-can > proximity_threshold              -> far_use_length
  near       eef-to-can <= proximity_threshold, not lifted -> near_use_length
  post_lift  can_z > lift_threshold (latched after near)   -> far_use_length

The lift latch is gated on having entered ``near`` first so the can's
above-table spawn drop doesn't trip it.

Usage::

    isaaclab.sh -p scripts/imitation_learning/eval_lingbot_isaac_a2_ee_ik_adaptive.py \\
        --websocket_host 172.18.1.26 --websocket_port 8007 \\
        --task "place the can in the tray" --episodes 1 --max_steps 5000 \\
        --far_use_length 50 --near_use_length 1 \\
        --proximity_threshold 0.20 --lift_threshold 1.05 --enable_cameras
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

parser = argparse.ArgumentParser(description="LingBot-VLA EE+IK eval with adaptive chunk reuse.")
parser.add_argument("--task_id", type=str, default="Isaac-PickPlace-A2-Abs-v0")
parser.add_argument("--task", type=str, default="place the can in the tray")
parser.add_argument("--websocket_host", type=str, default="172.18.1.26")
parser.add_argument("--websocket_port", type=int, default=8007)
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument("--max_steps", type=int, default=400)
parser.add_argument("--use_length", type=int, default=1,
                    help="Fallback chunk reuse if --far_use_length is 0.")
parser.add_argument("--far_use_length", type=int, default=50,
                    help="Chunk reuse when EE is farther than --proximity_threshold from the can, "
                         "and after the can has been lifted. 0 disables adaptive switching.")
parser.add_argument("--near_use_length", type=int, default=1,
                    help="Chunk reuse when EE is within --proximity_threshold and the can hasn't yet lifted.")
parser.add_argument("--proximity_threshold", type=float, default=0.20,
                    help="EE-to-can distance (m) at which to switch from far to near.")
parser.add_argument("--lift_threshold", type=float, default=1.05,
                    help="Latch the lifted state once the can z exceeds this (env frame). <=0 disables.")
parser.add_argument("--posture_cost", type=float, default=-1.0,
                    help="If >=0, widen the env's NullSpacePostureTask to all 14 arm joints + waist "
                         "and set its cost to this value. Target stays at init pose (no VLA driving).")
parser.add_argument("--finger_gate_distance", type=float, default=-1.0,
                    help="If >0, override the 24 hand joint targets to 0 (open hand) whenever the "
                         "EE-to-can distance exceeds this value. 0 disables the gate.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--lerobot_path", type=str, default="/home/wagner/code/lingbot-vla")
parser.add_argument("--debug_dump_every", type=int, default=0)
parser.add_argument("--debug_dir", type=str, default="/tmp/lingbot_isaac_eval_ee_ik_adaptive")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.num_envs != 1:
    raise SystemExit("Only num_envs=1 is supported.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

from isaaclab.controllers.pink_ik import NullSpacePostureTask

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

sys.path.insert(0, args_cli.lerobot_path)
from deploy.websocket_client_policy import WebsocketClientPolicy  # noqa: E402


_HAND_GROUPS = [
    [0, 10],
    [1, 11],
    [2, 12],
    [3, 13],
    [4],
    [14, 20, 22],
    [5, 15],
    [6, 16],
    [7, 17],
    [8, 18],
    [9],
    [19, 21, 23],
]


def fold_expand_hand_24(hand_24: np.ndarray) -> np.ndarray:
    out = hand_24.astype(np.float32, copy=True)
    for grp in _HAND_GROUPS:
        avg = float(np.mean(out[grp]))
        for k in grp:
            out[k] = avg
    return out


def model_action_to_env_action(action_53: np.ndarray) -> np.ndarray:
    assert action_53.shape == (53,), f"got {action_53.shape}"
    hand = fold_expand_hand_24(action_53[29:53])
    return np.concatenate([action_53[0:14], hand], axis=0)


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


def _compute_eef_to_can_and_z(env) -> tuple[float, float]:
    pol = env.unwrapped.observation_manager.compute()["policy"]
    le_pos = pol["left_eef_pos"][0].detach().cpu().numpy().astype(np.float32)
    re_pos = pol["right_eef_pos"][0].detach().cpu().numpy().astype(np.float32)
    obj_pos = pol["object_pos"][0].detach().cpu().numpy().astype(np.float32)
    return (
        float(min(np.linalg.norm(le_pos - obj_pos), np.linalg.norm(re_pos - obj_pos))),
        float(obj_pos[2]),
    )


_ARM_WAIST_NAMES = (
    [f"idx{13 + i:02d}_left_arm_joint{i + 1}" for i in range(7)]
    + [f"idx{20 + i:02d}_right_arm_joint{i + 1}" for i in range(7)]
    + ["waist_yaw_joint"]
)


def _widen_posture_task(cfg, cost: float) -> bool:
    actions = cfg.actions
    for attr_name, term_cfg in actions.__dict__.items():
        controller = getattr(term_cfg, "controller", None)
        if controller is None:
            continue
        tasks = getattr(controller, "variable_input_tasks", None)
        if not tasks:
            continue
        for task in tasks:
            if isinstance(task, NullSpacePostureTask):
                task.controlled_joints = list(_ARM_WAIST_NAMES)
                task.cost = cost
                print(
                    f"[eval] widened {attr_name}.NullSpacePostureTask: cost={cost} "
                    f"controlled_joints={task.controlled_joints}"
                )
                return True
    return False


def main():
    cfg = parse_env_cfg(args_cli.task_id, device=args_cli.device, num_envs=args_cli.num_envs)
    cfg.teleop_devices.devices = {}
    if hasattr(cfg, "recorders"):
        cfg.recorders = None
    cfg.episode_length_s = max(args_cli.max_steps * 0.05 + 5.0, getattr(cfg, "episode_length_s", 0.0))

    if args_cli.posture_cost >= 0.0:
        if not _widen_posture_task(cfg, args_cli.posture_cost):
            print("[eval] WARN: no NullSpacePostureTask found; --posture_cost ignored.")

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
            policy.reset("isaac_a2_ee_ik_adaptive")
        except Exception as e:
            print(f"  policy.reset warning: {e}")

        env.reset()
        chunk = None
        success = False
        lifted_latch = False
        was_near = False
        prev_phase: str | None = None

        for step in range(args_cli.max_steps):
            payload = encode_obs_for_policy(env, args_cli.task)
            if args_cli.debug_dump_every and step % args_cli.debug_dump_every == 0:
                np.save(debug_dir / f"ep{ep:03d}_step{step:04d}_state.npy",
                        payload["observation.state"])

            need_proximity = args_cli.far_use_length > 0 or args_cli.finger_gate_distance > 0
            eef_to_can = -1.0
            can_z = -1.0
            if need_proximity:
                eef_to_can, can_z = _compute_eef_to_can_and_z(env)

            if args_cli.far_use_length > 0:
                if eef_to_can <= args_cli.proximity_threshold:
                    was_near = True
                if (
                    args_cli.lift_threshold > 0
                    and was_near
                    and can_z > args_cli.lift_threshold
                ):
                    lifted_latch = True
                if lifted_latch:
                    use_len = args_cli.far_use_length
                    phase = "post_lift"
                elif eef_to_can <= args_cli.proximity_threshold:
                    use_len = args_cli.near_use_length
                    phase = "near"
                else:
                    use_len = args_cli.far_use_length
                    phase = "approach"
                if phase != prev_phase:
                    print(f"  step {step}: phase={phase} use_len={use_len} "
                          f"eef_to_can={eef_to_can:.3f} can_z={can_z:.3f}")
                    prev_phase = phase
            else:
                use_len = args_cli.use_length

            if step % max(1, use_len) == 0:
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
            if args_cli.finger_gate_distance > 0 and eef_to_can > args_cli.finger_gate_distance:
                env_row[14:38] = 0.0
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
