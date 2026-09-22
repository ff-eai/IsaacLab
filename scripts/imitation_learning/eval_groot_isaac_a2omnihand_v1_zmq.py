"""GR00T v1 (ZMQ+msgpack, port 5555) eval against the A2-OmniHand env.

The 5555 server uses the original sim-trained GR00T-N1.7 modality:
  state keys  = ["joints" (53-d), "left_eef" (9-d), "right_eef" (9-d)]
  video keys  = ["cam_high", "cam_chest_left", "cam_chest_right"]
  action keys = ["left_eef", "right_eef", "left_arm", "right_arm", "waist", "hands"]
                left/right_eef: relative xyz+rot6d (server decodes back to absolute
                                using the state we send)
                left/right_arm: absolute joint positions (7-d each)
                waist: absolute (1-d)
                hands: absolute s6_hand joints (24-d) — INCOMPATIBLE with OmniHand,
                       so we drop them and freeze the hand at its current pos.

This script drives the OmniHand env via its 46-d Pink-IK action term:
    action[ 0: 3] = left  EEF pos     ← decoded from action.left_eef[t]
    action[ 3: 7] = left  EEF quat
    action[ 7:10] = right EEF pos
    action[10:14] = right EEF quat
    action[14:46] = current 32-d hand state (frozen each tick)
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

parser = argparse.ArgumentParser(description="GR00T v1/ZMQ eval for OmniHand env.")
parser.add_argument("--task_id", type=str, default="Isaac-PickPlace-A2OmniHand-Abs-v0")
parser.add_argument("--task", type=str, default="place the coca-cola bottle in the tray")
parser.add_argument("--groot_host", type=str, default="172.18.1.26")
parser.add_argument("--groot_port", type=int, default=5555)
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument("--max_steps", type=int, default=1500)
parser.add_argument("--use_length", type=int, default=8,
                    help="Number of chunk rows applied per fetch (replan cadence).")
parser.add_argument("--chunk_start_row", type=int, default=4,
                    help="Skip the first N anchored rows of each chunk before "
                         "applying. Effective rows applied = "
                         "[chunk_start_row, chunk_start_row + use_length).")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--print_debug_every", type=int, default=16)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.num_envs != 1:
    raise SystemExit("Only num_envs=1 is supported.")
if args_cli.chunk_start_row + args_cli.use_length > 16:
    raise SystemExit("chunk_start_row + use_length > 16")

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
# msgpack <-> ndarray (mirrors gr00t/policy/server_client.py)
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


class GrootClient:
    def __init__(self, host: str, port: int, timeout_ms: int = 30000):
        self.context = zmq.Context()
        self.host, self.port, self.timeout_ms = host, port, timeout_ms
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
        self.socket.send(msgpack.packb(request, default=_encode_custom))
        msg = self.socket.recv()
        if msg == b"ERROR":
            raise RuntimeError("Server error.")
        resp = msgpack.unpackb(msg, object_hook=_decode_custom)
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

    def get_action(self, observation: dict, options: dict | None = None):
        resp = self.call("get_action", {"observation": observation, "options": options})
        return tuple(resp)


# ---------------------------------------------------------------------------
# Joint orderings
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
    # 24 hand joints (s6_hand naming — likely missing in OmniHand env)
    "L_index_1_joint", "L_middle_1_joint", "L_pinky_1_joint", "L_ring_1_joint",
    "L_thumb_swing_joint", "R_index_1_joint", "R_middle_1_joint", "R_pinky_1_joint",
    "R_ring_1_joint", "R_thumb_swing_joint", "L_index_2_joint", "L_middle_2_joint",
    "L_pinky_2_joint", "L_ring_2_joint", "L_thumb_1_joint", "R_index_2_joint",
    "R_middle_2_joint", "R_pinky_2_joint", "R_ring_2_joint", "R_thumb_1_joint",
    "L_thumb_2_joint", "R_thumb_2_joint", "L_thumb_3_joint", "R_thumb_3_joint",
]
assert len(GROOT_STATE_JOINT_NAMES) == 53

# Full 32-d _HAND_JOINTS list from pickplace_a2_omnihand_env_cfg.py — the env's
# Pink-IK action term consumes hand targets in this order.
OMNIHAND_HAND_JOINT_NAMES = [
    "L_thumb_roll_joint", "L_thumb_abad_joint", "L_thumb_mcp_joint",
    "L_index_abad_joint", "L_index_pip_joint",
    "L_middle_pip_joint",
    "L_ring_abad_joint", "L_ring_pip_joint",
    "L_pinky_abad_joint", "L_pinky_pip_joint",
    "L_thumb_pip_joint", "L_thumb_dip_joint",
    "L_index_dip_joint", "L_middle_dip_joint", "L_ring_dip_joint", "L_pinky_dip_joint",
    "R_thumb_roll_joint", "R_thumb_abad_joint", "R_thumb_mcp_joint",
    "R_index_abad_joint", "R_index_pip_joint",
    "R_middle_pip_joint",
    "R_ring_abad_joint", "R_ring_pip_joint",
    "R_pinky_abad_joint", "R_pinky_pip_joint",
    "R_thumb_pip_joint", "R_thumb_dip_joint",
    "R_index_dip_joint", "R_middle_dip_joint", "R_ring_dip_joint", "R_pinky_dip_joint",
]
assert len(OMNIHAND_HAND_JOINT_NAMES) == 32


# ---------------------------------------------------------------------------
# Quaternion / rot6d helpers
# ---------------------------------------------------------------------------
def _img_hwc(t: torch.Tensor) -> np.ndarray:
    return t[0].detach().to(torch.uint8).cpu().numpy()


def _quat_wxyz_to_rot6d(q: np.ndarray) -> np.ndarray:
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
    return q / max(np.linalg.norm(q), 1e-12)


def decode_eef_pose_from_action(eef9: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos = eef9[0:3].astype(np.float32)
    quat = _matrix_to_quat_wxyz(_rot6d_to_matrix(eef9[3:9]))
    return pos, quat


# ---------------------------------------------------------------------------
# Observation encoding (v1 modality)
# ---------------------------------------------------------------------------
def encode_obs_for_groot_v1(env, joint_pos_env: np.ndarray, env_to_groot: np.ndarray,
                            task: str) -> dict:
    pol = env.unwrapped.observation_manager.compute()["policy"]
    head_rgb = _img_hwc(pol["head_camera_rgb"])
    chest_l = _img_hwc(pol["chest_left_camera_rgb"])
    chest_r = _img_hwc(pol["chest_right_camera_rgb"])

    joints_53 = joint_pos_env[env_to_groot].astype(np.float32)
    le_pos = pol["left_eef_pos"][0].detach().cpu().numpy().astype(np.float32)
    le_quat = pol["left_eef_quat"][0].detach().cpu().numpy().astype(np.float32)
    re_pos = pol["right_eef_pos"][0].detach().cpu().numpy().astype(np.float32)
    re_quat = pol["right_eef_quat"][0].detach().cpu().numpy().astype(np.float32)
    le = np.concatenate([le_pos, _quat_wxyz_to_rot6d(le_quat)], axis=0).astype(np.float32)
    re = np.concatenate([re_pos, _quat_wxyz_to_rot6d(re_quat)], axis=0).astype(np.float32)

    return {
        "video": {
            "cam_high":        head_rgb[None, None, ...].astype(np.uint8),
            "cam_chest_left":  chest_l[None, None, ...].astype(np.uint8),
            "cam_chest_right": chest_r[None, None, ...].astype(np.uint8),
        },
        "state": {
            "joints":   joints_53[None, None, :],
            "left_eef":  le[None, None, :],
            "right_eef": re[None, None, :],
        },
        "language": {
            "annotation.human.task_description": [[task]],
        },
    }


def main():
    cfg = parse_env_cfg(args_cli.task_id, device=args_cli.device, num_envs=args_cli.num_envs)
    cfg.teleop_devices.devices = {}
    if hasattr(cfg, "recorders"):
        cfg.recorders = None
    cfg.episode_length_s = max(args_cli.max_steps * 0.05 + 5.0, getattr(cfg, "episode_length_s", 0.0))

    print(f"[eval] making env {args_cli.task_id} on {args_cli.device}")
    env = gym.make(args_cli.task_id, cfg=cfg).unwrapped
    env.seed = args_cli.seed

    env_joint_names = list(env.scene["robot"].data.joint_names)
    env_act_dim = env.action_manager.total_action_dim
    print(f"[eval] env articulation has {len(env_joint_names)} joints, action_dim={env_act_dim}")
    if env_act_dim != 46:
        print(f"[eval] WARN: expected 46-d Pink-IK action (OmniHand); got {env_act_dim}")

    env_name_to_idx = {n: i for i, n in enumerate(env_joint_names)}
    # Map GR00T 53-d state ordering -> env joint indices (skip missing).
    groot_to_env, missing = [], []
    for name in GROOT_STATE_JOINT_NAMES:
        if name in env_name_to_idx:
            groot_to_env.append(env_name_to_idx[name])
        else:
            groot_to_env.append(-1)
            missing.append(name)
    if missing:
        print(f"[eval] {len(missing)} groot state-joints missing in OmniHand env "
              f"(s6_hand finger joints — sent as 0 in state.joints):")
        print(f"  {missing}")
    # For the missing s6 hand joints, send zero in the joints[i] slot.
    env_to_groot_idx = np.asarray(
        [i if i >= 0 else 0 for i in groot_to_env], dtype=np.int64
    )
    is_real = np.asarray([i >= 0 for i in groot_to_env], dtype=bool)

    # 32-d hand env indices for the Pink-IK trailing slot.
    hand32_env_idx = np.asarray(
        [env_name_to_idx[n] for n in OMNIHAND_HAND_JOINT_NAMES], dtype=np.int64
    )

    print(f"[eval] connecting to GR00T at {args_cli.groot_host}:{args_cli.groot_port}")
    client = GrootClient(args_cli.groot_host, args_cli.groot_port)
    if not client.ping():
        raise SystemExit("[eval] ping to GR00T server failed")

    successes, episode_lengths = 0, []
    for ep in range(args_cli.episodes):
        env.reset()
        eef_chunk = None
        success = False
        for step in range(args_cli.max_steps):
            cur_joint_pos = env.scene["robot"].data.joint_pos[0].detach().cpu().numpy().astype(np.float32)
            joints_53 = cur_joint_pos[env_to_groot_idx]
            joints_53 = np.where(is_real, joints_53, 0.0).astype(np.float32)
            payload = encode_obs_for_groot_v1(env, joints_53, env_to_groot_idx, args_cli.task)

            if step % max(1, args_cli.use_length) == 0:
                t0 = time.time()
                try:
                    resp = client.get_action(payload)
                    action_dict = resp[0] if isinstance(resp, tuple) else resp
                except Exception as e:
                    print(f"  step {step}: ZMQ get_action failed: {e!r}")
                    break

                def _ax(name):
                    a = np.asarray(action_dict[name], dtype=np.float32)
                    return a[0] if a.ndim == 3 else a

                la = _ax("left_eef")    # (T, 9) absolute pos+rot6d (server already decoded)
                ra = _ax("right_eef")   # (T, 9)
                T = la.shape[0]
                eef_chunk = np.zeros((T, 14), dtype=np.float32)
                for t in range(T):
                    l_pos, l_quat = decode_eef_pose_from_action(la[t])
                    r_pos, r_quat = decode_eef_pose_from_action(ra[t])
                    eef_chunk[t, 0:3]  = l_pos
                    eef_chunk[t, 3:7]  = l_quat
                    eef_chunk[t, 7:10] = r_pos
                    eef_chunk[t, 10:14] = r_quat
                print(f"  step {step}: eef_chunk={eef_chunk.shape} infer={time.time() - t0:.2f}s")

            row_idx = args_cli.chunk_start_row + (step % args_cli.use_length)
            row = eef_chunk[row_idx]
            hand_now32 = cur_joint_pos[hand32_env_idx]
            action_vec = np.concatenate([row, hand_now32], axis=0).astype(np.float32)
            action_t = torch.from_numpy(action_vec).to(args_cli.device).unsqueeze(0)

            if args_cli.print_debug_every and step % args_cli.print_debug_every == 0:
                pol = env.unwrapped.observation_manager.compute()["policy"]
                cur_le = pol["left_eef_pos"][0].detach().cpu().numpy()
                cur_re = pol["right_eef_pos"][0].detach().cpu().numpy()
                tgt_l, tgt_r = row[0:3], row[7:10]
                print(
                    f"  [dbg s{step:04d}] "
                    f"L cur={cur_le.round(3).tolist()} → tgt={tgt_l.round(3).tolist()} "
                    f"|Δ|={float(np.linalg.norm(tgt_l - cur_le)):.4f}m  "
                    f"R cur={cur_re.round(3).tolist()} → tgt={tgt_r.round(3).tolist()} "
                    f"|Δ|={float(np.linalg.norm(tgt_r - cur_re)):.4f}m"
                )

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
