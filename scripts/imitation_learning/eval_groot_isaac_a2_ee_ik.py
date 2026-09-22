"""Closed-loop eval for the Nvidia GR00T N1.7 policy server on Isaac-PickPlace-A2.

Default ``--action_mode joint`` applies GR00T's ``left_arm`` / ``right_arm`` /
``waist`` / ``hands`` joint targets directly (same as ``eval_groot_isaac_a2.py``).
Use ``--action_mode eef_ik`` only if your checkpoint truly commands via EEF; that
path ignores arm joint outputs and often leaves arms frozen when the policy
predicts joint-space actions.

Action from GR00T (B=1, T=16 chunks):
    joint mode: left_arm[7] + right_arm[7] + waist[1] + hands[24] -> joint_pos targets
    eef_ik mode: left_eef[9] + right_eef[9] + hands[12 mapped] -> Pink IK (26-d).
                 ZMQ server :5560 already returns absolute EEF; use --decode_relative_eef
                 only for raw relative outputs. left_arm/right_arm are ignored in eef_ik.

Usage::

    isaaclab.sh -p scripts/imitation_learning/eval_groot_isaac_a2_ee_ik.py \\
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

parser = argparse.ArgumentParser(description="Nvidia GR00T EE+IK eval inside Isaac Lab.")
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
parser.add_argument("--debug_dir", type=str, default="/tmp/groot_isaac_eval_ee_ik")
parser.add_argument("--ros2_pub", action="store_true",
                    help="Forward joint_pos_target after every env.step to a UDP socket so a "
                         "separate ROS2 bridge node can republish on /motion/control/*_joint_command.")
parser.add_argument("--ros2_pub_host", default="127.0.0.1")
parser.add_argument("--ros2_pub_port", type=int, default=51234)
parser.add_argument("--dummy_joints_state", action="store_true",
                    help="Replace state.joints (53-d) in the observation payload with zeros. "
                         "Cameras + state.left/right_eef remain real. Diagnostic for whether the "
                         "model relies on the joint state.")
parser.add_argument("--side", choices=("both", "left", "right"), default="both",
                    help="Which arm + hand should track the model's commands. The other side is "
                         "frozen at its measured EE pose / hand joint values each tick.")
parser.add_argument(
    "--action_mode",
    choices=("joint", "eef_ik"),
    default="joint",
    help="joint: drive arms/hands/waist from GR00T joint outputs (recommended). "
         "eef_ik: Pink IK from server EEF poses only (ignores arm joint outputs).",
)
parser.add_argument(
    "--decode_relative_eef",
    action="store_true",
    help="eef_ik only: apply GR00T T_ref @ T_rel on left_eef/right_eef. Default off because "
         "the policy server already unapplies relative EEF to absolute before ZMQ return.",
)
parser.add_argument("--disable_waist", action="store_true",
                    help="eef_ik only: drop waist_yaw_joint from Pink IK. joint mode: skip waist targets.")
parser.add_argument("--print_legs_every", type=int, default=10,
                    help="Print left+right leg joint positions every N env steps (0 = off).")
parser.add_argument("--print_arms_every", type=int, default=10,
                    help="Print right-arm joint targets vs current every N steps (0 = off).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.num_envs != 1:
    raise SystemExit("Only num_envs=1 is supported.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import msgpack
import numpy as np
import socket
import struct
import torch
import zmq

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.pick_place.pickplace_a2_env_cfg import (
    enable_pickplace_a2_cameras,
)
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.utils import configclass
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
# Joint ordering (from training meta/info.json)
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
    "L_index_1_joint", "L_middle_1_joint", "L_pinky_1_joint", "L_ring_1_joint",
    "L_thumb_swing_joint", "R_index_1_joint", "R_middle_1_joint", "R_pinky_1_joint",
    "R_ring_1_joint", "R_thumb_swing_joint", "L_index_2_joint", "L_middle_2_joint",
    "L_pinky_2_joint", "L_ring_2_joint", "L_thumb_1_joint", "R_index_2_joint",
    "R_middle_2_joint", "R_pinky_2_joint", "R_ring_2_joint", "R_thumb_1_joint",
    "L_thumb_2_joint", "R_thumb_2_joint", "L_thumb_3_joint", "R_thumb_3_joint",
]
assert len(GROOT_STATE_JOINT_NAMES) == 53

# GR00T action.hands is 24-d; Pink-IK env action uses 12 (_HAND_JOINTS in pickplace_a2_env_cfg).
GROOT_HA_TO_ENV_HAND_IDX = np.array(
    [
        0, 4, 14, 1, 3, 2,  # left: index, thumb_swing, thumb_1, middle, ring, pinky
        5, 9, 19, 6, 8, 7,  # right
    ],
    dtype=np.int64,
)
assert len(GROOT_HA_TO_ENV_HAND_IDX) == 12
PINK_IK_ACTION_DIM = 26  # 2 * (pos3 + quat4) + 12 hand joints

ACTION_LEFT_ARM_NAMES = [f"idx{13 + i:02d}_left_arm_joint{i + 1}" for i in range(7)]
ACTION_RIGHT_ARM_NAMES = [f"idx{20 + i:02d}_right_arm_joint{i + 1}" for i in range(7)]
ACTION_WAIST_NAMES = ["waist_yaw_joint"]
ACTION_HAND_NAMES = GROOT_STATE_JOINT_NAMES[29:]
assert len(ACTION_HAND_NAMES) == 24
# s6_hand in Isaac-PickPlace-A2-Abs (matches pickplace_a2_env_cfg._HAND_JOINTS)
ENV_HAND_JOINT_NAMES = [
    "L_index_1_joint", "L_thumb_swing_joint", "L_thumb_1_joint",
    "L_middle_1_joint", "L_ring_1_joint", "L_pinky_1_joint",
    "R_index_1_joint", "R_thumb_swing_joint", "R_thumb_1_joint",
    "R_middle_1_joint", "R_ring_1_joint", "R_pinky_1_joint",
]
assert len(ENV_HAND_JOINT_NAMES) == 12


@configclass
class _JointActOnlyCfg:
    joint_action: JointPositionActionCfg = JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=1.0,
        use_default_offset=False,
    )


# Joint name order expected by the ROS2 bridge (a2_joint_bridge.py).
# Wire format: 14 arm + 1 waist + 24 hand = 39 float32, little-endian.
_ROS2_ARM_NAMES = [
    "idx13_left_arm_joint1", "idx14_left_arm_joint2", "idx15_left_arm_joint3",
    "idx16_left_arm_joint4", "idx17_left_arm_joint5", "idx18_left_arm_joint6",
    "idx19_left_arm_joint7",
    "idx20_right_arm_joint1", "idx21_right_arm_joint2", "idx22_right_arm_joint3",
    "idx23_right_arm_joint4", "idx24_right_arm_joint5", "idx25_right_arm_joint6",
    "idx26_right_arm_joint7",
]
_ROS2_WAIST_NAMES = ["waist_yaw_joint"]
_ROS2_HAND_NAMES = [
    "L_index_1_joint", "L_middle_1_joint", "L_pinky_1_joint", "L_ring_1_joint",
    "L_thumb_swing_joint", "R_index_1_joint", "R_middle_1_joint", "R_pinky_1_joint",
    "R_ring_1_joint", "R_thumb_swing_joint",
    "L_index_2_joint", "L_middle_2_joint", "L_pinky_2_joint", "L_ring_2_joint",
    "L_thumb_1_joint", "R_index_2_joint", "R_middle_2_joint", "R_pinky_2_joint",
    "R_ring_2_joint", "R_thumb_1_joint",
    "L_thumb_2_joint", "R_thumb_2_joint", "L_thumb_3_joint", "R_thumb_3_joint",
]
_ROS2_PACK_FMT = "<39f"


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


def _rot6d_to_matrix(r6: np.ndarray) -> np.ndarray:
    """Decode Groot rot6d (first two rows of R, flattened) back to a 3x3 matrix
    via Gram-Schmidt."""
    row0 = np.asarray(r6[0:3], dtype=np.float64)
    row1 = np.asarray(r6[3:6], dtype=np.float64)
    n0 = row0 / max(np.linalg.norm(row0), 1e-12)
    row1_proj = row1 - n0 * float(np.dot(n0, row1))
    n1 = row1_proj / max(np.linalg.norm(row1_proj), 1e-12)
    n2 = np.cross(n0, n1)
    return np.stack([n0, n1, n2], axis=0)  # 3x3, rows = row0,row1,row2


def _matrix_to_rot6d(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> GR00T rot6d (first two rows flattened)."""
    return R[:2, :].reshape(-1).astype(np.float32)


def _homogeneous_from_eef9(eef9: np.ndarray) -> np.ndarray:
    """eef9 = xyz(3) + rot6d(6) -> 4x4 SE(3) homogeneous matrix."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _rot6d_to_matrix(eef9[3:9])
    T[:3, 3] = eef9[0:3].astype(np.float64)
    return T


def _eef9_from_homogeneous(T: np.ndarray) -> np.ndarray:
    """4x4 homogeneous matrix -> eef9 = xyz(3) + rot6d(6)."""
    pos = T[:3, 3].astype(np.float32)
    return np.concatenate([pos, _matrix_to_rot6d(T[:3, :3])], axis=0).astype(np.float32)


def relative_eef9_to_absolute_eef9(rel9: np.ndarray, ref9: np.ndarray) -> np.ndarray:
    """GR00T N1.7 relative EEF -> absolute: T_abs = T_ref @ T_rel (XYZ_ROT6D)."""
    T_abs = _homogeneous_from_eef9(ref9) @ _homogeneous_from_eef9(rel9)
    return _eef9_from_homogeneous(T_abs)


def _matrix_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> (w,x,y,z) unit quaternion. Standard branchful form."""
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


def encode_obs_for_groot(env, joint_pos_env: np.ndarray, env_to_groot: np.ndarray, task: str) -> dict:
    pol = env.unwrapped.observation_manager.compute()["policy"]
    head_rgb = _img_hwc(pol["head_camera_rgb"])
    chest_l = _img_hwc(pol["chest_left_camera_rgb"])
    chest_r = _img_hwc(pol["chest_right_camera_rgb"])

    joints_53 = joint_pos_env[env_to_groot].astype(np.float32)
    if args_cli.dummy_joints_state:
        joints_53 = np.zeros_like(joints_53)
    le_pos = pol["left_eef_pos"][0].detach().cpu().numpy().astype(np.float32)
    le_quat = pol["left_eef_quat"][0].detach().cpu().numpy().astype(np.float32)
    re_pos = pol["right_eef_pos"][0].detach().cpu().numpy().astype(np.float32)
    re_quat = pol["right_eef_quat"][0].detach().cpu().numpy().astype(np.float32)
    le_state = np.concatenate([le_pos, _quat_wxyz_to_rot6d(le_quat)], axis=0).astype(np.float32)
    re_state = np.concatenate([re_pos, _quat_wxyz_to_rot6d(re_quat)], axis=0).astype(np.float32)

    return {
        "video": {
            "cam_high": head_rgb[None, None, ...].astype(np.uint8),
            "cam_chest_left": chest_l[None, None, ...].astype(np.uint8),
            "cam_chest_right": chest_r[None, None, ...].astype(np.uint8),
        },
        "state": {
            "joints": joints_53[None, None, :],
            "left_eef": le_state[None, None, :],
            "right_eef": re_state[None, None, :],
        },
        "language": {
            "annotation.human.task_description": [[task]],
        },
    }


def decode_eef_pose_from_action(eef9: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decode absolute eef9 (xyz + rot6d) to pos(3) and quat wxyz(4)."""
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

    if args_cli.action_mode == "joint":
        cfg.actions = _JointActOnlyCfg()
        print("[eval] action_mode=joint (left_arm/right_arm/waist/hands -> joint_pos targets)")
    else:
        rel = "T_ref @ T_rel" if args_cli.decode_relative_eef else "off (server absolute EEF)"
        print(
            f"[eval] action_mode=eef_ik (Pink IK; EEF decode: {rel}; "
            "left_arm/right_arm/waist ignored — use --action_mode joint for arm joint commands)"
        )

    # Isaac-PickPlace-A2-Abs-v0 disables scene cameras by default; GR00T needs RGB.
    if "OmniHand" not in args_cli.task_id and getattr(cfg.scene, "head_camera", None) is None:
        enable_pickplace_a2_cameras(cfg)
        print("[eval] enabled head + chest RGB cameras for GR00T video observations")

    if args_cli.disable_waist and args_cli.action_mode == "eef_ik":
        # Drop waist_yaw_joint from each Pink IK action term's controlled set so
        # the IK no longer touches the waist. The hand action term (separate
        # JointPositionAction within the same upper_body_ik term) is unchanged.
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
    if args_cli.action_mode == "eef_ik" and env_act_dim != PINK_IK_ACTION_DIM:
        print(f"  WARN: expected {PINK_IK_ACTION_DIM}-d Pink-IK action input; got {env_act_dim}")

    env_name_to_idx = {n: i for i, n in enumerate(env_joint_names)}

    def _env_idx(names):
        out, miss = [], []
        for n in names:
            if n in env_name_to_idx:
                out.append(env_name_to_idx[n])
            else:
                miss.append(n)
        if miss:
            raise SystemExit(f"[eval] joints missing in env articulation: {miss}")
        return np.asarray(out, dtype=np.int64)

    left_arm_env_idx = right_arm_env_idx = waist_env_idx = hand_env_idx = None
    if args_cli.action_mode == "joint":
        left_arm_env_idx = _env_idx(ACTION_LEFT_ARM_NAMES)
        right_arm_env_idx = _env_idx(ACTION_RIGHT_ARM_NAMES)
        waist_env_idx = _env_idx(ACTION_WAIST_NAMES)
        hand_env_idx = _env_idx(ENV_HAND_JOINT_NAMES)
        print(
            f"[eval] joint targets: arms {len(left_arm_env_idx)}+{len(right_arm_env_idx)} "
            f"waist={len(waist_env_idx)} hands={len(hand_env_idx)}"
        )
    groot_to_env = []
    missing = []
    for name in GROOT_STATE_JOINT_NAMES:
        if name in env_name_to_idx:
            groot_to_env.append(env_name_to_idx[name])
        else:
            groot_to_env.append(-1)
            missing.append(name)
    if missing:
        print(f"[eval] WARN groot joints missing in env: {missing}")
    env_to_groot = np.asarray(groot_to_env, dtype=np.int64)

    # Resolve leg-joint indices in env order for periodic logging.
    _LEG_PATTERNS = ("hip_roll", "hip_yaw", "hip_pitch", "tarsus", "toe_pitch", "toe_roll")
    leg_env_idx = [(i, n) for i, n in enumerate(env_joint_names)
                   if any(p in n for p in _LEG_PATTERNS)]
    print(f"[eval] leg joints (env order): {[n for _, n in leg_env_idx]}")

    ros2_sock = None
    ros2_arm_idx = ros2_waist_idx = ros2_hand_idx = None
    if args_cli.ros2_pub:
        ros2_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        ros2_dest = (args_cli.ros2_pub_host, args_cli.ros2_pub_port)
        def _idx_list(names):
            out = []
            miss = []
            for n in names:
                if n in env_name_to_idx:
                    out.append(env_name_to_idx[n])
                else:
                    miss.append(n)
            if miss:
                print(f"[eval] WARN ROS2 bridge joints missing in env: {miss}")
            return np.asarray(out, dtype=np.int64)
        ros2_arm_idx = _idx_list(_ROS2_ARM_NAMES)
        ros2_waist_idx = _idx_list(_ROS2_WAIST_NAMES)
        ros2_hand_idx = _idx_list(_ROS2_HAND_NAMES)
        print(f"[eval] ROS2 publish enabled -> udp://{ros2_dest[0]}:{ros2_dest[1]} "
              f"(arm={len(ros2_arm_idx)} waist={len(ros2_waist_idx)} hand={len(ros2_hand_idx)})")

    print(f"[eval] connecting to GR00T at {args_cli.groot_host}:{args_cli.groot_port}")
    client = GrootClient(args_cli.groot_host, args_cli.groot_port)
    if not client.ping():
        raise SystemExit("[eval] ping to GR00T server failed")

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
        chunk = None
        success = False
        for step in range(args_cli.max_steps):
            cur_joint_pos = env.scene["robot"].data.joint_pos[0].detach().cpu().numpy().astype(np.float32)
            if args_cli.print_legs_every > 0 and step % args_cli.print_legs_every == 0:
                vals = ", ".join(f"{n}={float(cur_joint_pos[i]):+.4f}" for i, n in leg_env_idx)
                print(f"  step {step} legs: {vals}")
            payload = encode_obs_for_groot(env, cur_joint_pos, env_to_groot, args_cli.task)
            if args_cli.debug_dump_every and step % args_cli.debug_dump_every == 0:
                np.save(debug_dir / f"ep{ep:03d}_step{step:04d}_state.npy",
                        np.concatenate([payload["state"]["joints"][0, 0],
                                        payload["state"]["left_eef"][0, 0],
                                        payload["state"]["right_eef"][0, 0]]))

            if step % max(1, args_cli.use_length) == 0:
                t0 = time.time()
                action_dict, info = client.get_action(payload)

                def _ax(name):
                    a = np.asarray(action_dict[name], dtype=np.float32)
                    return a[0] if a.ndim == 3 else a  # -> (T, D)

                if args_cli.action_mode == "joint":
                    la = _ax("left_arm")
                    ra = _ax("right_arm")
                    wa = _ax("waist")
                    ha = _ax("hands")
                    T_action = la.shape[0]
                    chunk = np.concatenate([la, ra, wa, ha], axis=1)  # (T, 39)
                    assert chunk.shape[1] == 39, f"unexpected joint chunk width {chunk.shape[1]}"
                else:
                    la_eef = _ax("left_eef")
                    ra_eef = _ax("right_eef")
                    ha = _ax("hands")
                    T_action = la_eef.shape[0]
                    ref_le = payload["state"]["left_eef"][0, 0].astype(np.float32)
                    ref_re = payload["state"]["right_eef"][0, 0].astype(np.float32)
                    rows = np.zeros((T_action, PINK_IK_ACTION_DIM), dtype=np.float32)
                    for t in range(T_action):
                        l_eef9 = la_eef[t]
                        r_eef9 = ra_eef[t]
                        if args_cli.decode_relative_eef:
                            l_eef9 = relative_eef9_to_absolute_eef9(l_eef9, ref_le)
                            r_eef9 = relative_eef9_to_absolute_eef9(r_eef9, ref_re)
                        l_pos, l_quat = decode_eef_pose_from_action(l_eef9)
                        r_pos, r_quat = decode_eef_pose_from_action(r_eef9)
                        rows[t, 0:3] = l_pos
                        rows[t, 3:7] = l_quat
                        rows[t, 7:10] = r_pos
                        rows[t, 10:14] = r_quat
                        rows[t, 14:26] = ha[t][GROOT_HA_TO_ENV_HAND_IDX]
                    chunk = rows
                    if args_cli.print_arms_every > 0 and step % args_cli.print_arms_every == 0:
                        t0_dbg = 0
                        l9 = la_eef[t0_dbg].copy()
                        r9 = ra_eef[t0_dbg].copy()
                        if args_cli.decode_relative_eef:
                            l9 = relative_eef9_to_absolute_eef9(l9, ref_le)
                            r9 = relative_eef9_to_absolute_eef9(r9, ref_re)
                        l_tgt, _ = decode_eef_pose_from_action(l9)
                        r_tgt, _ = decode_eef_pose_from_action(r9)
                        pol = env.unwrapped.observation_manager.compute()["policy"]
                        cur_le = pol["left_eef_pos"][0].detach().cpu().numpy()
                        cur_re = pol["right_eef_pos"][0].detach().cpu().numpy()
                        ra0 = _ax("right_arm")[t0_dbg]
                        print(
                            f"  step {step} eef t0: "
                            f"L |srv-state|={float(np.linalg.norm(la_eef[t0_dbg, :3] - ref_le[:3])):.4f} "
                            f"|tgt-cur|={float(np.linalg.norm(l_tgt - cur_le)):.4f}m  "
                            f"R |srv-state|={float(np.linalg.norm(ra_eef[t0_dbg, :3] - ref_re[:3])):.4f} "
                            f"|tgt-cur|={float(np.linalg.norm(r_tgt - cur_re)):.4f}m  "
                            f"(ignored) right_arm_j1 tgt={float(ra0[0]):+.4f}"
                        )
                print(f"  step {step}: chunk shape={chunk.shape} infer={time.time() - t0:.2f}s")

            row = chunk[step % chunk.shape[0]].copy()

            if args_cli.action_mode == "joint":
                target = cur_joint_pos.copy()
                if args_cli.side in ("both", "left"):
                    target[left_arm_env_idx] = row[0:7]
                if args_cli.side in ("both", "right"):
                    target[right_arm_env_idx] = row[7:14]
                if not args_cli.disable_waist and args_cli.side == "both":
                    target[waist_env_idx] = row[14:15]
                ha_row = row[15:39]
                if args_cli.side == "both":
                    target[hand_env_idx] = ha_row[GROOT_HA_TO_ENV_HAND_IDX]
                elif args_cli.side == "right":
                    target[hand_env_idx[6:12]] = ha_row[GROOT_HA_TO_ENV_HAND_IDX][6:12]
                elif args_cli.side == "left":
                    target[hand_env_idx[0:6]] = ha_row[GROOT_HA_TO_ENV_HAND_IDX][0:6]
                action_t = torch.from_numpy(target).to(args_cli.device).unsqueeze(0)
                if args_cli.print_arms_every > 0 and step % args_cli.print_arms_every == 0:
                    j = int(right_arm_env_idx[0])
                    print(
                        f"  step {step} right_arm_j1: cur={cur_joint_pos[j]:+.4f} "
                        f"target={target[j]:+.4f} delta={target[j] - cur_joint_pos[j]:+.4f}"
                    )
            else:
                if args_cli.side != "both":
                    pol = env.unwrapped.observation_manager.compute()["policy"]
                    hand_state = pol["hand_joint_state"][0].detach().cpu().numpy().astype(np.float32)
                    if args_cli.side == "right":
                        row[0:3] = pol["left_eef_pos"][0].detach().cpu().numpy().astype(np.float32)
                        row[3:7] = pol["left_eef_quat"][0].detach().cpu().numpy().astype(np.float32)
                        row[14:20] = hand_state[0:6]
                    else:
                        row[7:10] = pol["right_eef_pos"][0].detach().cpu().numpy().astype(np.float32)
                        row[10:14] = pol["right_eef_quat"][0].detach().cpu().numpy().astype(np.float32)
                        row[20:26] = hand_state[6:12]
                action_t = torch.from_numpy(row).to(args_cli.device).unsqueeze(0)
            _, _, terminated, truncated, _ = env.step(action_t)
            if ros2_sock is not None:
                target_t = env.scene["robot"].data.joint_pos_target
                target = target_t[0].detach().cpu().numpy().astype(np.float32)
                packet = struct.pack(
                    _ROS2_PACK_FMT,
                    *target[ros2_arm_idx],
                    *target[ros2_waist_idx],
                    *target[ros2_hand_idx],
                )
                try:
                    ros2_sock.sendto(packet, ros2_dest)
                except OSError as e:
                    print(f"  step {step}: ros2 udp send failed: {e}")
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
