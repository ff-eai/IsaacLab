"""Closed-loop eval for a Nvidia GR00T policy server trained with right-arm-only
obs and action against the IsaacLab A2 pick-place task.

Server modality (queried via get_modality_config on 172.18.1.26:5557):
    state:  right_arm[7], right_hand[12], right_eef[9]
    action: right_eef[9]   (RELATIVE EEF XYZ_ROT6D, decoded server-side)
            right_arm[7]   (ABSOLUTE NON_EEF)
            right_hands[12](ABSOLUTE NON_EEF)
    video:  cam_high, cam_chest_left, cam_chest_right
    language.annotation.human.task_description

Action chunks have 16 steps (delta_indices 0..15). The env runs the standard
Pink-IK 38-d action (14 EE pos+quat + 24 hands). On each tick this eval
populates the right wrist + right-hand slice with the model's command and
freezes the left wrist + left-hand slice at the current measured state.

Usage::

    isaaclab.sh -p scripts/imitation_learning/eval_groot_isaac_a2_right_only.py \\
        --groot_host 172.18.1.26 --groot_port 5557 \\
        --task "place the can in the tray" --episodes 1 --max_steps 5000 \\
        --use_length 16 --enable_cameras
"""

from __future__ import annotations

import argparse
import functools
import io
import os
import time
from pathlib import Path

print = functools.partial(print, flush=True)  # noqa: A001
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import pinocchio  # noqa: F401  (must come before AppLauncher)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Right-only Nvidia GR00T eval inside Isaac Lab.")
parser.add_argument("--task_id", type=str, default="Isaac-PickPlace-A2-Abs-v0")
parser.add_argument("--task", type=str, default="place the can in the tray")
parser.add_argument("--groot_host", type=str, default="172.18.1.26")
parser.add_argument("--groot_port", type=int, default=5557)
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument("--max_steps", type=int, default=400)
parser.add_argument("--use_length", type=int, default=16,
                    help="Re-query GR00T every N env steps. Action chunks have 16 steps.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--debug_dump_every", type=int, default=0)
parser.add_argument("--debug_dir", type=str, default="/tmp/groot_isaac_eval_right_only")
parser.add_argument("--disable_waist", action="store_true",
                    help="Drop waist_yaw_joint from the env's Pink IK controlled joint set.")
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
        return tuple(resp)

    def get_modality_config(self):
        return self.call("get_modality_config", requires_input=False)


# ---------------------------------------------------------------------------
# Joint orderings for the right-only modality. Right-arm and right-hand orders
# follow the natural slice of the parent 53-d state used at training time.
# ---------------------------------------------------------------------------
RIGHT_ARM_NAMES = [f"idx{20 + i:02d}_right_arm_joint{i + 1}" for i in range(7)]
assert len(RIGHT_ARM_NAMES) == 7

RIGHT_HAND_NAMES = [
    "R_index_1_joint", "R_middle_1_joint", "R_pinky_1_joint", "R_ring_1_joint", "R_thumb_swing_joint",
    "R_index_2_joint", "R_middle_2_joint", "R_pinky_2_joint", "R_ring_2_joint", "R_thumb_1_joint",
    "R_thumb_2_joint", "R_thumb_3_joint",
]
assert len(RIGHT_HAND_NAMES) == 12

# Indices into the env's 24-d Pink-IK hand action term (order baked into
# pickplace_a2_env_cfg._HAND_JOINTS) that correspond to RIGHT_HAND_NAMES.
_LEFT_HAND_IDX_24  = np.array([0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 20, 22], dtype=np.int64)
_RIGHT_HAND_IDX_24 = np.array([5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 21, 23], dtype=np.int64)


def _img_hwc(t: torch.Tensor) -> np.ndarray:
    return t[0].detach().to(torch.uint8).cpu().numpy()


def _quat_wxyz_to_rot6d(q: np.ndarray) -> np.ndarray:
    """wxyz quat -> first two ROWS of the rotation matrix, flattened (Groot rot6d)."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array([
        1 - 2 * (y * y + z * z),
        2 * (x * y - z * w),
        2 * (x * z + y * w),
        2 * (x * y + z * w),
        1 - 2 * (x * x + z * z),
        2 * (y * z - x * w),
    ], dtype=np.float32)


def _rot6d_to_matrix(r6: np.ndarray) -> np.ndarray:
    row0 = np.asarray(r6[0:3], dtype=np.float64)
    row1 = np.asarray(r6[3:6], dtype=np.float64)
    n0 = row0 / max(np.linalg.norm(row0), 1e-12)
    row1_proj = row1 - n0 * float(np.dot(n0, row1))
    n1 = row1_proj / max(np.linalg.norm(row1_proj), 1e-12)
    n2 = np.cross(n0, n1)
    return np.stack([n0, n1, n2], axis=0)


def _matrix_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    tr = m00 + m11 + m22
    if tr > 0.0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float32)
    n = np.linalg.norm(q)
    return q / max(n, 1e-12)


def encode_obs_for_groot(env, joint_pos_env: np.ndarray,
                         right_arm_env_idx: np.ndarray,
                         right_hand_env_idx: np.ndarray,
                         task: str) -> dict:
    pol = env.unwrapped.observation_manager.compute()["policy"]
    head_rgb = _img_hwc(pol["head_camera_rgb"])
    chest_l = _img_hwc(pol["chest_left_camera_rgb"])
    chest_r = _img_hwc(pol["chest_right_camera_rgb"])

    right_arm = joint_pos_env[right_arm_env_idx].astype(np.float32)         # (7,)
    right_hand = joint_pos_env[right_hand_env_idx].astype(np.float32)       # (12,)

    re_pos = pol["right_eef_pos"][0].detach().cpu().numpy().astype(np.float32)
    re_quat = pol["right_eef_quat"][0].detach().cpu().numpy().astype(np.float32)
    re_state = np.concatenate([re_pos, _quat_wxyz_to_rot6d(re_quat)], axis=0).astype(np.float32)

    return {
        "video": {
            "cam_high": head_rgb[None, None, ...].astype(np.uint8),
            "cam_chest_left": chest_l[None, None, ...].astype(np.uint8),
            "cam_chest_right": chest_r[None, None, ...].astype(np.uint8),
        },
        "state": {
            "right_arm": right_arm[None, None, :],
            "right_hand": right_hand[None, None, :],
            "right_eef": re_state[None, None, :],
        },
        "language": {
            "annotation.human.task_description": [[task]],
        },
    }


def decode_eef_pose_from_action(eef9: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos = eef9[0:3].astype(np.float32)
    R = _rot6d_to_matrix(eef9[3:9])
    quat = _matrix_to_quat_wxyz(R)
    return pos, quat


def main():
    cfg = parse_env_cfg(args_cli.task_id, device=args_cli.device, num_envs=args_cli.num_envs)
    cfg.teleop_devices.devices = {}
    if hasattr(cfg, "recorders"):
        cfg.recorders = None
    cfg.episode_length_s = max(args_cli.max_steps * 0.05 + 5.0, getattr(cfg, "episode_length_s", 0.0))

    if args_cli.disable_waist:
        for attr_name, term_cfg in cfg.actions.__dict__.items():
            joints = getattr(term_cfg, "pink_controlled_joint_names", None)
            if joints is None:
                continue
            new_joints = [j for j in joints if j != "waist_yaw_joint"]
            if len(new_joints) != len(joints):
                term_cfg.pink_controlled_joint_names = new_joints
                print(f"[eval] disabled waist on {attr_name}: pink_controlled_joint_names "
                      f"now {len(new_joints)} joints (was {len(joints)})")

    print(f"[eval] making env {args_cli.task_id} on {args_cli.device}")
    env = gym.make(args_cli.task_id, cfg=cfg).unwrapped
    env.seed = args_cli.seed

    env_joint_names = list(env.scene["robot"].data.joint_names)
    n_joints = len(env_joint_names)
    env_act_dim = env.action_manager.total_action_dim
    print(f"[eval] env articulation has {n_joints} joints, action_dim={env_act_dim}")
    if env_act_dim != 38:
        print(f"  WARN: expected 38-d Pink-IK action input; got {env_act_dim}")

    env_name_to_idx = {n: i for i, n in enumerate(env_joint_names)}

    def _idx(names):
        out, miss = [], []
        for n in names:
            if n in env_name_to_idx:
                out.append(env_name_to_idx[n])
            else:
                miss.append(n)
        if miss:
            raise SystemExit(f"[eval] joints missing in env articulation: {miss}")
        return np.asarray(out, dtype=np.int64)

    right_arm_env_idx = _idx(RIGHT_ARM_NAMES)
    right_hand_env_idx = _idx(RIGHT_HAND_NAMES)
    print(f"[eval] right_arm env idx ({len(right_arm_env_idx)}): {right_arm_env_idx.tolist()}")
    print(f"[eval] right_hand env idx ({len(right_hand_env_idx)}): {right_hand_env_idx.tolist()}")

    print(f"[eval] connecting to GR00T at {args_cli.groot_host}:{args_cli.groot_port}")
    client = GrootClient(args_cli.groot_host, args_cli.groot_port)
    if not client.ping():
        raise SystemExit("[eval] ping to GR00T server failed")
    try:
        modality = client.get_modality_config()
        if isinstance(modality, dict):
            print(f"[eval] server modality keys: {list(modality.keys())}")
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
        chunk = None  # (T, 38)
        success = False
        for step in range(args_cli.max_steps):
            cur_joint_pos = env.scene["robot"].data.joint_pos[0].detach().cpu().numpy().astype(np.float32)
            payload = encode_obs_for_groot(env, cur_joint_pos,
                                           right_arm_env_idx, right_hand_env_idx,
                                           args_cli.task)
            if args_cli.debug_dump_every and step % args_cli.debug_dump_every == 0:
                np.save(debug_dir / f"ep{ep:03d}_step{step:04d}_state.npy",
                        np.concatenate([payload["state"]["right_arm"][0, 0],
                                        payload["state"]["right_hand"][0, 0],
                                        payload["state"]["right_eef"][0, 0]]))

            if step % max(1, args_cli.use_length) == 0:
                t0 = time.time()
                action_dict, info = client.get_action(payload)
                if action_dict is None:
                    print(f"  step {step}: server returned None — abort")
                    break

                def _ax(name):
                    a = np.asarray(action_dict[name], dtype=np.float32)
                    return a[0] if a.ndim == 3 else a  # -> (T, D)

                ra_eef = _ax("right_eef")    # (T, 9)  pos+rot6d, server-decoded -> absolute
                ha = _ax("right_hands")      # (T, 12) right hand qpos targets
                T_action = ra_eef.shape[0]
                # Build env Pink-IK action chunk (38-d). Left half is filled with
                # the *current* measured pose+hand each tick (below); right half
                # comes from the model.
                rows = np.zeros((T_action, 38), dtype=np.float32)
                for t in range(T_action):
                    r_pos, r_quat = decode_eef_pose_from_action(ra_eef[t])
                    rows[t, 7:10] = r_pos
                    rows[t, 10:14] = r_quat
                    rows[t, 14 + _RIGHT_HAND_IDX_24] = ha[t]
                chunk = rows
                print(f"  step {step}: chunk shape={chunk.shape} infer={time.time() - t0:.2f}s")

            row = chunk[step % chunk.shape[0]].copy()

            # Freeze left wrist + left hand at their measured values so Pink-IK
            # holds them still while the right side tracks the model.
            pol = env.unwrapped.observation_manager.compute()["policy"]
            row[0:3] = pol["left_eef_pos"][0].detach().cpu().numpy().astype(np.float32)
            row[3:7] = pol["left_eef_quat"][0].detach().cpu().numpy().astype(np.float32)
            hand_state = pol["hand_joint_state"][0].detach().cpu().numpy().astype(np.float32)
            row[14 + _LEFT_HAND_IDX_24] = hand_state[_LEFT_HAND_IDX_24]

            action_t = torch.from_numpy(row).to(args_cli.device).unsqueeze(0)
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
