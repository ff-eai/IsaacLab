"""Closed-loop eval for the Nvidia GR00T N1.7 policy server (Isaac-GR00T)
running at 172.18.1.26:5555 against the IsaacLab A2 pick-place task.

Server checkpoint metadata (read from
``/data/home/vincent.cui/GrootN17/runs/a2_pickplace_v1/.../experiment_cfg``):
  * embodiment_tag: new_embodiment
  * dataset:        a2_pickplace_v2_combined_ee  (LeRobot v2.1, FPS=30)
  * state shape 71 = joints[0:53] + left_eef[53:62] + right_eef[62:71]
                     left/right_eef = [pos_xyz(3) + rot6d(6)]
  * action shape 57 = left_eef[0:9]+right_eef[9:18] (relative xyz+rot6d) +
                     left_arm[18:25]+right_arm[25:32]+waist[32:33]+hands[33:57]
                     arm/waist/hands are absolute joint targets.

This eval uses ONLY the absolute joint slices (left_arm + right_arm + waist +
hands = 39 dims) to drive the env via JointPositionAction (same scheme as
``eval_lingbot_isaac_a2_joint_only.py``). The relative EE xyz+rot6d slices
are ignored to avoid having to decode the relative encoding.

Server protocol: ZMQ REQ + msgpack with custom encoder for ``np.ndarray``
(matches ``Isaac-GR00T/gr00t/policy/server_client.py``).

Usage::

    isaaclab.sh -p scripts/imitation_learning/eval_groot_isaac_a2.py \\
        --groot_host 172.18.1.26 --groot_port 5555 \\
        --task "place the can in the tray" --episodes 1 --max_steps 5000 \\
        --use_length 16 --enable_cameras
"""

from __future__ import annotations

import argparse
import functools
import io
import os
import sys
import time
from pathlib import Path

print = functools.partial(print, flush=True)  # noqa: A001
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import pinocchio  # noqa: F401  (must come before AppLauncher)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Nvidia GR00T eval inside Isaac Lab.")
parser.add_argument("--task_id", type=str, default="Isaac-PickPlace-A2-Abs-v0")
parser.add_argument("--task", type=str, default="place the can in the tray")
parser.add_argument("--groot_host", type=str, default="172.18.1.26")
parser.add_argument("--groot_port", type=int, default=5555)
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument("--max_steps", type=int, default=400)
parser.add_argument("--use_length", type=int, default=16,
                    help="Replan cadence: re-query GR00T every N env steps. Action chunks have 16 steps.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--debug_dump_every", type=int, default=0)
parser.add_argument("--debug_dir", type=str, default="/tmp/groot_isaac_eval")
parser.add_argument("--no_eef_state", action="store_true",
                    help="Drop state.left_eef and state.right_eef from the observation payload "
                         "(use this for models trained on joints-only state).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.num_envs != 1:
    raise SystemExit("Only num_envs=1 is supported.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import msgpack
import numpy as np
import torch
import zmq

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.utils import configclass

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


# ---------------------------------------------------------------------------
# msgpack <-> ndarray serialization (mirrors gr00t/policy/server_client.py)
# ---------------------------------------------------------------------------
def _encode_custom(obj):
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
    return obj


def _decode_custom(obj):
    if isinstance(obj, dict) and "__ndarray_class__" in obj:
        return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
    return obj


def _msg_to_bytes(data) -> bytes:
    return msgpack.packb(data, default=_encode_custom)


def _msg_from_bytes(b: bytes):
    return msgpack.unpackb(b, object_hook=_decode_custom)


class GrootClient:
    """Minimal ZMQ REQ client for the GR00T policy server."""

    def __init__(self, host: str, port: int, timeout_ms: int = 30000):
        self.context = zmq.Context()
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self._init_socket()

    def _init_socket(self):
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def call(self, endpoint: str, data: dict | None = None, requires_input: bool = True):
        request: dict = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data or {}
        try:
            self.socket.send(_msg_to_bytes(request))
            msg = self.socket.recv()
        except zmq.error.Again:
            self._init_socket()
            raise
        if msg == b"ERROR":
            raise RuntimeError("Server error.")
        resp = _msg_from_bytes(msg)
        if isinstance(resp, dict) and "error" in resp:
            raise RuntimeError(f"Server error: {resp['error']}")
        return resp

    def ping(self) -> bool:
        try:
            self.call("ping", requires_input=False)
            return True
        except Exception:
            self._init_socket()
            return False

    def reset(self, options: dict | None = None):
        return self.call("reset", {"options": options})

    def get_action(self, observation: dict, options: dict | None = None):
        resp = self.call("get_action", {"observation": observation, "options": options})
        return tuple(resp)  # list -> (action_dict, info_dict)

    def get_modality_config(self):
        return self.call("get_modality_config", requires_input=False)


# ---------------------------------------------------------------------------
# Joint ordering (from
# /data/home/vincent.cui/GrootN17/datasets_gr00t/.../meta/info.json)
# ---------------------------------------------------------------------------
GROOT_STATE_JOINT_NAMES = [
    "idx01_left_hip_roll", "idx07_right_hip_roll", "idx27_head_joint1", "waist_yaw_joint",
    "idx02_left_hip_yaw", "idx08_right_hip_yaw", "idx28_head_joint2", "idx13_left_arm_joint1",
    "idx20_right_arm_joint1", "idx03_left_hip_pitch", "idx09_right_hip_pitch",
    "idx14_left_arm_joint2", "idx21_right_arm_joint2", "idx04_left_tarsus", "idx10_right_tarsus",
    "idx15_left_arm_joint3", "idx22_right_arm_joint3", "idx05_left_toe_pitch",
    "idx11_right_toe_pitch", "idx16_left_arm_joint4", "idx23_right_arm_joint4",
    "idx06_left_toe_roll", "idx12_right_toe_roll", "idx17_left_arm_joint5",
    "idx24_right_arm_joint5", "idx18_left_arm_joint6", "idx25_right_arm_joint6",
    "idx19_left_arm_joint7", "idx26_right_arm_joint7",
    # 24 hand joints
    "L_index_1_joint", "L_middle_1_joint", "L_pinky_1_joint", "L_ring_1_joint",
    "L_thumb_swing_joint", "R_index_1_joint", "R_middle_1_joint", "R_pinky_1_joint",
    "R_ring_1_joint", "R_thumb_swing_joint", "L_index_2_joint", "L_middle_2_joint",
    "L_pinky_2_joint", "L_ring_2_joint", "L_thumb_1_joint", "R_index_2_joint",
    "R_middle_2_joint", "R_pinky_2_joint", "R_ring_2_joint", "R_thumb_1_joint",
    "L_thumb_2_joint", "R_thumb_2_joint", "L_thumb_3_joint", "R_thumb_3_joint",
]
assert len(GROOT_STATE_JOINT_NAMES) == 53

# action.left_arm[i] -> idx13+i_left_arm_joint(i+1)  (i=0..6)
# action.right_arm[i] -> idx20+i_right_arm_joint(i+1)
# action.waist[0] -> waist_yaw_joint
# action.hands[i] -> 24-name list above (positions 29..52)
ACTION_LEFT_ARM_NAMES = [f"idx{13 + i:02d}_left_arm_joint{i + 1}" for i in range(7)]
ACTION_RIGHT_ARM_NAMES = [f"idx{20 + i:02d}_right_arm_joint{i + 1}" for i in range(7)]
ACTION_WAIST_NAMES = ["waist_yaw_joint"]
ACTION_HAND_NAMES = GROOT_STATE_JOINT_NAMES[29:]
assert len(ACTION_HAND_NAMES) == 24


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


def _quat_wxyz_to_rot6d(q: np.ndarray) -> np.ndarray:
    """wxyz quat -> 6d as Groot encodes it: R[:2, :].flatten() (first two ROWS)."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array([
        1 - 2 * (y * y + z * z),  # R00
        2 * (x * y - z * w),      # R01
        2 * (x * z + y * w),      # R02
        2 * (x * y + z * w),      # R10
        1 - 2 * (x * x + z * z),  # R11
        2 * (y * z - x * w),      # R12
    ], dtype=np.float32)


def encode_obs_for_groot(env, joint_pos_env: np.ndarray, env_to_groot: np.ndarray, task: str) -> dict:
    pol = env.unwrapped.observation_manager.compute()["policy"]
    head_rgb = _img_hwc(pol["head_camera_rgb"])
    chest_l = _img_hwc(pol["chest_left_camera_rgb"])
    chest_r = _img_hwc(pol["chest_right_camera_rgb"])

    # Reorder env joint_pos -> dataset (groot) joint order
    joints_53 = joint_pos_env[env_to_groot].astype(np.float32)

    le_pos = pol["left_eef_pos"][0].detach().cpu().numpy().astype(np.float32)
    le_quat = pol["left_eef_quat"][0].detach().cpu().numpy().astype(np.float32)
    re_pos = pol["right_eef_pos"][0].detach().cpu().numpy().astype(np.float32)
    re_quat = pol["right_eef_quat"][0].detach().cpu().numpy().astype(np.float32)
    le_state = np.concatenate([le_pos, _quat_wxyz_to_rot6d(le_quat)], axis=0).astype(np.float32)
    re_state = np.concatenate([re_pos, _quat_wxyz_to_rot6d(re_quat)], axis=0).astype(np.float32)

    # Shapes per gr00t_policy.check_observation:
    #   video: (B=1, T=1, H, W, C) uint8
    #   state: (B=1, T=1, D) float32
    #   language: list[list[str]]  shape (B=1, T=1)
    state = {
        "joints": joints_53[None, None, :],
    }
    if not args_cli.no_eef_state:
        state["left_eef"] = le_state[None, None, :]
        state["right_eef"] = re_state[None, None, :]
    return {
        "video": {
            "cam_high": head_rgb[None, None, ...].astype(np.uint8),
            "cam_chest_left": chest_l[None, None, ...].astype(np.uint8),
            "cam_chest_right": chest_r[None, None, ...].astype(np.uint8),
        },
        "state": state,
        "language": {
            "annotation.human.task_description": [[task]],
        },
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

    env_joint_names = list(env.scene["robot"].data.joint_names)
    n_joints = len(env_joint_names)
    print(f"[eval] env articulation has {n_joints} joints, action_dim={env.action_manager.total_action_dim}")

    # env joint index -> groot state index
    env_name_to_idx = {n: i for i, n in enumerate(env_joint_names)}
    groot_name_to_env_idx = []
    missing_groot = []
    for name in GROOT_STATE_JOINT_NAMES:
        if name in env_name_to_idx:
            groot_name_to_env_idx.append(env_name_to_idx[name])
        else:
            groot_name_to_env_idx.append(-1)
            missing_groot.append(name)
    if missing_groot:
        print(f"[eval] WARN groot joints missing in env articulation: {missing_groot}")
    env_to_groot = np.asarray(groot_name_to_env_idx, dtype=np.int64)

    # action driver names -> env joint indices for the 39 driven joints
    def env_idx_of(name: str) -> int:
        return env_name_to_idx.get(name, -1)

    left_arm_env_idx = np.asarray([env_idx_of(n) for n in ACTION_LEFT_ARM_NAMES], dtype=np.int64)
    right_arm_env_idx = np.asarray([env_idx_of(n) for n in ACTION_RIGHT_ARM_NAMES], dtype=np.int64)
    waist_env_idx = np.asarray([env_idx_of(n) for n in ACTION_WAIST_NAMES], dtype=np.int64)
    hand_env_idx = np.asarray([env_idx_of(n) for n in ACTION_HAND_NAMES], dtype=np.int64)
    assert (left_arm_env_idx >= 0).all() and (right_arm_env_idx >= 0).all() \
        and (waist_env_idx >= 0).all() and (hand_env_idx >= 0).all(), \
        "action joint name(s) missing in env articulation"
    print(f"[eval] action joint mapping: arms={len(left_arm_env_idx)}+{len(right_arm_env_idx)} "
          f"waist={len(waist_env_idx)} hands={len(hand_env_idx)} = "
          f"{len(left_arm_env_idx)+len(right_arm_env_idx)+len(waist_env_idx)+len(hand_env_idx)} driven")

    print(f"[eval] connecting to GR00T at {args_cli.groot_host}:{args_cli.groot_port}")
    client = GrootClient(args_cli.groot_host, args_cli.groot_port)
    if not client.ping():
        raise SystemExit("[eval] ping to GR00T server failed")
    try:
        modality = client.get_modality_config()
        print(f"[eval] server modality keys: {list(modality.keys()) if isinstance(modality, dict) else modality}")
    except Exception as e:
        print(f"[eval] get_modality_config warning: {e}")

    debug_dir = Path(args_cli.debug_dir)
    if args_cli.debug_dump_every:
        debug_dir.mkdir(parents=True, exist_ok=True)

    successes, episode_lengths = 0, []
    for ep in range(args_cli.episodes):
        try:
            client.reset()
        except Exception as e:
            print(f"  client.reset warning: {e}")

        env.reset()
        chunk = None  # (T, 39) absolute joint targets
        success = False
        for step in range(args_cli.max_steps):
            cur_joint_pos = env.scene["robot"].data.joint_pos[0].detach().cpu().numpy().astype(np.float32)
            payload = encode_obs_for_groot(env, cur_joint_pos, env_to_groot, args_cli.task)
            if args_cli.debug_dump_every and step % args_cli.debug_dump_every == 0:
                np.save(debug_dir / f"ep{ep:03d}_step{step:04d}_state.npy",
                        np.concatenate([payload["state.joints"][0],
                                        payload["state.left_eef"][0],
                                        payload["state.right_eef"][0]]))

            if step % max(1, args_cli.use_length) == 0:
                t0 = time.time()
                action_dict, info = client.get_action(payload)
                if action_dict is None:
                    print(f"  step {step}: server returned None — abort")
                    break
                # gr00t_policy.check_action expects shape (B, T, D); for B=1 we
                # squeeze. Action keys come from training modality_configs:
                # left_eef, right_eef, left_arm, right_arm, waist, hands.
                def _ax(name):
                    a = np.asarray(action_dict[name], dtype=np.float32)
                    return a[0] if a.ndim == 3 else a  # -> (T, D)
                la = _ax("left_arm")
                ra = _ax("right_arm")
                wa = _ax("waist")
                ha = _ax("hands")
                T_action = la.shape[0]
                chunk = np.concatenate([la, ra, wa, ha], axis=1)  # (T, 39)
                assert chunk.shape == (T_action, 39), f"unexpected chunk shape {chunk.shape}"
                print(f"  step {step}: chunk shape={chunk.shape} infer={time.time() - t0:.2f}s")

            row = chunk[step % chunk.shape[0]]
            target = cur_joint_pos.copy()
            target[left_arm_env_idx] = row[0:7]
            target[right_arm_env_idx] = row[7:14]
            target[waist_env_idx] = row[14:15]
            target[hand_env_idx] = row[15:39]
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
