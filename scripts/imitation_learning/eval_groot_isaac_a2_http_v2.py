"""Closed-loop eval against the v2 HTTP GR00T policy server (port 18001).

The v2 server (`/mnt/extreme_pro/Vincent/GrootN17/policy_server.py`) loads a
GR00T policy trained on the real-robot A2_task_6139 dataset. Its modality
config is **different** from the sim-trained 5555/5556 servers:

  state.modality_keys  = ["effector", "arm"]
  action.modality_keys = ["effector", "arm"]
  video.modality_keys  = ["head_front_color", "hand_left", "hand_right"]

  effector: 20-d finger encoder values (real-robot OmniHand)
  arm:      14-d joint positions (state/joint/position[314:328])

  action.effector: ABSOLUTE finger encoder targets
  action.arm:      RELATIVE joint deltas (per N1.7 recipe)

Wire format: HTTP POST /act, pickled body, pickled response.

Sim-side caveats:
  * effector 20-d encoders have no clean mapping to our 24 URDF finger joints
    in sim. We send ZEROS for state.effector (OOD but model still drives arm).
  * action.effector is dropped — we hold the env's hand at its measured state.
  * Cameras: head_camera -> head_front_color (resized 1280x720 -> 640x360),
    left_wrist -> hand_left, right_wrist -> hand_right (already 640x480).
"""

from __future__ import annotations

import argparse
import functools
import http.client
import os
import pickle
import sys
import time
from pathlib import Path

print = functools.partial(print, flush=True)  # noqa: A001
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import pinocchio  # noqa: F401  (must come before AppLauncher)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="GR00T v2 HTTP eval inside Isaac Lab.")
parser.add_argument("--task_id", type=str, default="Isaac-PickPlace-A2-Abs-v0")
parser.add_argument("--task", type=str, default="place the can in the tray")
parser.add_argument("--groot_host", type=str, default="172.18.1.26")
parser.add_argument("--groot_port", type=int, default=18001)
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument("--max_steps", type=int, default=400)
parser.add_argument("--use_length", type=int, default=16,
                    help="Replan after consuming this many rows of the chunk (max 16).")
parser.add_argument("--chunk_start_row", type=int, default=0,
                    help="Skip the first N rows of every fetched chunk before applying it. "
                         "Useful because the model anchors row 0 at the current pose; row N "
                         "carries N*step-of-displacement, so larger N = larger commanded "
                         "motion per env step. Effective rows applied are "
                         "[chunk_start_row, chunk_start_row + use_length). Must satisfy "
                         "chunk_start_row + use_length <= 16.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--debug_dump_every", type=int, default=0)
parser.add_argument("--debug_dir", type=str, default="/tmp/groot_isaac_eval_v2")
parser.add_argument("--print_debug_every", type=int, default=0,
                    help="Every N env steps print compact obs/action snapshot to stdout. "
                         "Shows current arm/EEF vs model-commanded targets so you can spot "
                         "frame mismatches, near-zero deltas, etc. 0 disables.")
parser.add_argument("--use_chest_cameras", action="store_true",
                    help="Switch to the OmniHand sim-trained schema (port 18003 server): "
                         "video keys chest_left/chest_right (not hand_left/hand_right), "
                         "state adds left_eef/right_eef (9-d each: pos+rot6d) and a live 20-d "
                         "effector built from the 20 active OmniHand joints, and action.effector "
                         "is written back to those 20 joints as ABSOLUTE targets.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.num_envs != 1:
    raise SystemExit("Only num_envs=1 is supported.")

if args_cli.chunk_start_row + args_cli.use_length > 16:
    raise SystemExit(
        f"chunk_start_row({args_cli.chunk_start_row}) + use_length({args_cli.use_length}) "
        f"= {args_cli.chunk_start_row + args_cli.use_length} > 16 (model chunk size)"
    )
if args_cli.chunk_start_row < 0 or args_cli.use_length < 1:
    raise SystemExit("chunk_start_row must be >=0 and use_length >=1")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import cv2
import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.utils import configclass

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


# ---------------------------------------------------------------------------
# v2 model expects these 14 arm joints (state/joint/position[314:328]) — order
# is the canonical A2 left-then-right arm joint sequence used in the dataset.
# Same 14 names exist in our sim URDF.
# ---------------------------------------------------------------------------
ARM_JOINT_NAMES = [
    "idx13_left_arm_joint1", "idx14_left_arm_joint2", "idx15_left_arm_joint3",
    "idx16_left_arm_joint4", "idx17_left_arm_joint5", "idx18_left_arm_joint6",
    "idx19_left_arm_joint7",
    "idx20_right_arm_joint1", "idx21_right_arm_joint2", "idx22_right_arm_joint3",
    "idx23_right_arm_joint4", "idx24_right_arm_joint5", "idx25_right_arm_joint6",
    "idx26_right_arm_joint7",
]
ARM_DIM = 14
EFFECTOR_DIM = 20  # 10 active joints per hand × 2; URDF-active order [L 10, R 10]
T_VIDEO = 1
T_ACTION = 16

# 20 OmniHand active joints in URDF active-joint order (mimic children excluded).
# Matches `_HAND_JOINTS[0:10]` (left) + `_HAND_JOINTS[16:26]` (right) from
# `pickplace_a2_omnihand_env_cfg.py`.
ACTIVE_HAND_JOINT_NAMES = [
    "L_thumb_roll_joint", "L_thumb_abad_joint", "L_thumb_mcp_joint",
    "L_index_abad_joint", "L_index_pip_joint",
    "L_middle_pip_joint",
    "L_ring_abad_joint", "L_ring_pip_joint",
    "L_pinky_abad_joint", "L_pinky_pip_joint",
    "R_thumb_roll_joint", "R_thumb_abad_joint", "R_thumb_mcp_joint",
    "R_index_abad_joint", "R_index_pip_joint",
    "R_middle_pip_joint",
    "R_ring_abad_joint", "R_ring_pip_joint",
    "R_pinky_abad_joint", "R_pinky_pip_joint",
]
assert len(ACTIVE_HAND_JOINT_NAMES) == EFFECTOR_DIM


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


def _resize_rgb(img: np.ndarray, w: int, h: int) -> np.ndarray:
    if img.shape[0] == h and img.shape[1] == w:
        return img
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def _quat_wxyz_to_rot6d(q: np.ndarray) -> np.ndarray:
    """wxyz quat -> 6d (first two ROWS of rotation matrix), matching Groot dataset."""
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
    """Decode rot6d (first two rows of R, flattened) → 3x3 matrix via Gram-Schmidt."""
    row0 = np.asarray(r6[0:3], dtype=np.float64)
    row1 = np.asarray(r6[3:6], dtype=np.float64)
    n0 = row0 / max(np.linalg.norm(row0), 1e-12)
    row1_proj = row1 - n0 * float(np.dot(n0, row1))
    n1 = row1_proj / max(np.linalg.norm(row1_proj), 1e-12)
    n2 = np.cross(n0, n1)
    return np.stack([n0, n1, n2], axis=0)


def _matrix_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix → (w,x,y,z) unit quaternion."""
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
    """eef9 = pos(3) + rot6d(6)  ->  (pos(3), quat_wxyz(4))."""
    pos = eef9[0:3].astype(np.float32)
    quat = _matrix_to_quat_wxyz(_rot6d_to_matrix(eef9[3:9]))
    return pos, quat


# Full 32-d _HAND_JOINTS list from pickplace_a2_omnihand_env_cfg.py — the Pink-IK
# action term consumes hand targets in this order in its trailing 32-d slot.
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

# Per-joint URDF limits for the 20 active OmniHand joints, ordered to match
# ACTIVE_HAND_JOINT_NAMES (L 10 + R 10). Used to translate between sim joint
# radians and the model's [0, 4000] encoder counts:
#   encoder_i = clip((rad_i - lower_i) / (upper_i - lower_i), 0, 1) * 4000
#   rad_i     = encoder_i / 4000 * (upper_i - lower_i) + lower_i
# Approximation: assumes linear, encoder=0 ↔ lower limit, encoder=4000 ↔ upper.
# The OmniHand README warns the real hardware uses NONLINEAR coupling for the
# mimic chain; the 20 active joints are 1-to-1 with their encoder so this is
# reasonable for them. Calibration constants below come from
# omnihand_description-omnihandT2_1/assets/urdf/omnihand_right.urdf (left has
# identical limits).
EFFECTOR_FULL_RANGE = 4000.0
_HAND_LIMITS_LOWER_PER_HAND = np.array([
    -0.0297,   # thumb_roll
    -1.6424,   # thumb_abad
     0.0,      # thumb_mcp
    -0.1641,   # index_abad
     0.0,      # index_pip
     0.0,      # middle_pip
     0.0,      # ring_abad
     0.0,      # ring_pip
     0.0,      # pinky_abad
     0.0,      # pinky_pip
], dtype=np.float32)
_HAND_LIMITS_UPPER_PER_HAND = np.array([
    1.1214,
    0.0454,
    0.8416,
    0.0,
    1.4835,
    1.4835,
    0.1693,
    1.4835,
    0.1850,
    1.4835,
], dtype=np.float32)
_HAND_LIMITS_LOWER = np.concatenate(
    [_HAND_LIMITS_LOWER_PER_HAND, _HAND_LIMITS_LOWER_PER_HAND]
)
_HAND_LIMITS_UPPER = np.concatenate(
    [_HAND_LIMITS_UPPER_PER_HAND, _HAND_LIMITS_UPPER_PER_HAND]
)
_HAND_LIMITS_RANGE = _HAND_LIMITS_UPPER - _HAND_LIMITS_LOWER
assert _HAND_LIMITS_LOWER.shape == (EFFECTOR_DIM,)


def encode_hand_to_encoder(rad_20: np.ndarray) -> np.ndarray:
    """rad_20 (radians, length 20) → encoder counts [0, 4000]."""
    n = (rad_20.astype(np.float32) - _HAND_LIMITS_LOWER) / _HAND_LIMITS_RANGE
    return np.clip(n, 0.0, 1.0) * EFFECTOR_FULL_RANGE


def decode_hand_from_encoder(enc_20: np.ndarray) -> np.ndarray:
    """encoder counts [0, 4000] (length 20) → rad."""
    n = np.clip(enc_20.astype(np.float32) / EFFECTOR_FULL_RANGE, 0.0, 1.0)
    return n * _HAND_LIMITS_RANGE + _HAND_LIMITS_LOWER


# Map a 20-d active-joint vector to the full 32-d _HAND_JOINTS layout, with the
# 12 mimic joints derived from active joints via URDF mimic multipliers.
# _HAND_JOINTS slots:
#   [ 0.. 9] L active,  [10..15] L mimic,  [16..25] R active,  [26..31] R mimic.
# Mimic table (URDF):
#   thumb_pip  = thumb_mcp  × 1.33
#   thumb_dip  = thumb_mcp  × 1.30
#   index_dip  = index_pip  × 1.097
#   middle_dip = middle_pip × 1.097
#   ring_dip   = ring_pip   × 1.097
#   pinky_dip  = pinky_pip  × 1.097
_FULL_TO_ACTIVE_IDX = np.array([
    0,  1,  2,  3,  4,  5,  6,  7,  8,  9,        # L active
    2,  2,  4,  5,  7,  9,                        # L mimic ← driving active idx
    10, 11, 12, 13, 14, 15, 16, 17, 18, 19,       # R active
    12, 12, 14, 15, 17, 19,                       # R mimic
], dtype=np.int64)
_FULL_MIMIC_MULT = np.array([
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.33, 1.30, 1.097, 1.097, 1.097, 1.097,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.33, 1.30, 1.097, 1.097, 1.097, 1.097,
], dtype=np.float32)
assert _FULL_TO_ACTIVE_IDX.shape == (32,)


def active20_to_full_hand32(active_20: np.ndarray) -> np.ndarray:
    """Expand 20 active joint angles to the 32-d _HAND_JOINTS layout (mimic via URDF)."""
    return (active_20.astype(np.float32)[_FULL_TO_ACTIVE_IDX] * _FULL_MIMIC_MULT)


def encode_obs_v2(env, arm_pos_14: np.ndarray, task: str,
                  hand_pos_20: np.ndarray | None = None) -> dict:
    pol = env.unwrapped.observation_manager.compute()["policy"]
    head = _img_hwc(pol["head_camera_rgb"])                      # 720x1280
    head_resized = _resize_rgb(head, 640, 360)                   # match dataset 360x640

    if args_cli.use_chest_cameras:
        # OmniHand sim-trained schema (port 18003): chest cameras + eef + live effector.
        chest_l = _img_hwc(pol["chest_left_camera_rgb"])         # 480x640
        chest_r = _img_hwc(pol["chest_right_camera_rgb"])        # 480x640
        le_pos = pol["left_eef_pos"][0].detach().cpu().numpy().astype(np.float32)
        le_quat = pol["left_eef_quat"][0].detach().cpu().numpy().astype(np.float32)
        re_pos = pol["right_eef_pos"][0].detach().cpu().numpy().astype(np.float32)
        re_quat = pol["right_eef_quat"][0].detach().cpu().numpy().astype(np.float32)
        le = np.concatenate([le_pos, _quat_wxyz_to_rot6d(le_quat)], axis=0)
        re = np.concatenate([re_pos, _quat_wxyz_to_rot6d(re_quat)], axis=0)
        # Convert sim joint pos (rad) → real-robot encoder counts [0, 4000].
        eff_in = (encode_hand_to_encoder(hand_pos_20) if hand_pos_20 is not None
                  else np.zeros(EFFECTOR_DIM, dtype=np.float32))
        return {
            "video": {
                "head_front_color": head_resized[None, None, ...].astype(np.uint8),
                "chest_left":       chest_l[None, None, ...].astype(np.uint8),
                "chest_right":      chest_r[None, None, ...].astype(np.uint8),
            },
            "state": {
                "effector": eff_in.astype(np.float32)[None, None, :],   # (1,1,20)
                "arm":      arm_pos_14.astype(np.float32)[None, None, :],  # (1,1,14)
                "left_eef":  le.astype(np.float32)[None, None, :],         # (1,1,9)
                "right_eef": re.astype(np.float32)[None, None, :],         # (1,1,9)
            },
            "language": {
                "annotation.human.task_description": [[task]],
            },
        }

    # Original v2 schema (real-robot wrist cameras, effector zeros).
    hand_l = _img_hwc(pol["left_wrist_camera_rgb"])              # 480x640
    hand_r = _img_hwc(pol["right_wrist_camera_rgb"])             # 480x640
    eff_zero = np.zeros(EFFECTOR_DIM, dtype=np.float32)
    return {
        "video": {
            "head_front_color": head_resized[None, None, ...].astype(np.uint8),
            "hand_left":        hand_l[None, None, ...].astype(np.uint8),
            "hand_right":       hand_r[None, None, ...].astype(np.uint8),
        },
        "state": {
            "effector": eff_zero[None, None, :],                 # (1,1,20)
            "arm":      arm_pos_14.astype(np.float32)[None, None, :],
        },
        "language": {
            "annotation.human.task_description": [[task]],
        },
    }


def post_act(host: str, port: int, payload: dict, timeout: float = 30.0) -> dict:
    body = pickle.dumps(payload)
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        conn.request("POST", "/act", body=body, headers={
            "Content-Type": "application/octet-stream",
            "Content-Length": str(len(body)),
        })
        resp = conn.getresponse()
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {resp.read()[:200]!r}")
        data = resp.read()
    finally:
        conn.close()
    return pickle.loads(data)


def main():
    cfg = parse_env_cfg(args_cli.task_id, device=args_cli.device, num_envs=args_cli.num_envs)
    if not args_cli.use_chest_cameras:
        # Original v2 path: bypass Pink-IK and feed full-articulation joint targets.
        cfg.actions = _JointActOnlyCfg()
    # When --use_chest_cameras: keep the env's default Pink-IK action (46-d for
    # OmniHand: 14-d EE pose + 32-d hand) and feed model.left_eef / right_eef as
    # absolute targets, freezing the hand at its current state.
    cfg.teleop_devices.devices = {}
    if hasattr(cfg, "recorders"):
        cfg.recorders = None
    cfg.episode_length_s = max(args_cli.max_steps * 0.05 + 5.0, getattr(cfg, "episode_length_s", 0.0))

    print(f"[eval] making env {args_cli.task_id} on {args_cli.device}")
    env = gym.make(args_cli.task_id, cfg=cfg).unwrapped
    env.seed = args_cli.seed

    env_joint_names = list(env.scene["robot"].data.joint_names)
    env_name_to_idx = {n: i for i, n in enumerate(env_joint_names)}
    arm_env_idx = np.asarray([env_name_to_idx[n] for n in ARM_JOINT_NAMES], dtype=np.int64)
    if args_cli.use_chest_cameras:
        missing = [n for n in ACTIVE_HAND_JOINT_NAMES if n not in env_name_to_idx]
        if missing:
            raise SystemExit(f"[eval] active hand joint(s) missing in env: {missing}")
        hand_env_idx = np.asarray(
            [env_name_to_idx[n] for n in ACTIVE_HAND_JOINT_NAMES], dtype=np.int64
        )
        # Full 32-d hand index (Pink-IK action's trailing slot consumes this order).
        missing32 = [n for n in OMNIHAND_HAND_JOINT_NAMES if n not in env_name_to_idx]
        if missing32:
            raise SystemExit(f"[eval] OmniHand joint(s) missing in env: {missing32}")
        hand32_env_idx = np.asarray(
            [env_name_to_idx[n] for n in OMNIHAND_HAND_JOINT_NAMES], dtype=np.int64
        )
    else:
        hand_env_idx = None
        hand32_env_idx = None
    print(f"[eval] env articulation has {len(env_joint_names)} joints, "
          f"action_dim={env.action_manager.total_action_dim}, arm_idx={arm_env_idx.tolist()}")
    if hand_env_idx is not None:
        print(f"[eval] hand_idx (20 active)={hand_env_idx.tolist()}")
        print(f"[eval] hand32_idx (full _HAND_JOINTS)={hand32_env_idx.tolist()}")
        if env.action_manager.total_action_dim != 46:
            print(f"[eval] WARN: expected 46-d Pink-IK action; got "
                  f"{env.action_manager.total_action_dim}")

    print(f"[eval] connecting to GR00T HTTP at {args_cli.groot_host}:{args_cli.groot_port}/act")

    debug_dir = Path(args_cli.debug_dir)
    if args_cli.debug_dump_every:
        debug_dir.mkdir(parents=True, exist_ok=True)

    successes, episode_lengths = 0, []
    for ep in range(args_cli.episodes):
        env.reset()
        arm_chunk = None        # (T, 14) absolute arm joint targets (joint-target path)
        eef_chunk = None        # (T, 14) [l_pos+l_quat | r_pos+r_quat] (EE-IK path)
        hand_chunk32 = None     # (T, 32) absolute hand-joint targets, full _HAND_JOINTS layout
        success = False
        for step in range(args_cli.max_steps):
            cur_joint_pos = env.scene["robot"].data.joint_pos[0].detach().cpu().numpy().astype(np.float32)
            arm_now = cur_joint_pos[arm_env_idx]
            hand_now20 = cur_joint_pos[hand_env_idx] if hand_env_idx is not None else None
            payload = encode_obs_v2(env, arm_now, args_cli.task, hand_pos_20=hand_now20)
            if args_cli.debug_dump_every and step % args_cli.debug_dump_every == 0:
                np.save(debug_dir / f"ep{ep:03d}_step{step:04d}_arm.npy", arm_now)

            if step % max(1, args_cli.use_length) == 0:
                t0 = time.time()
                try:
                    action_dict = post_act(args_cli.groot_host, args_cli.groot_port, payload)
                except Exception as e:
                    print(f"  step {step}: HTTP /act failed: {e!r}")
                    break

                def _ax(name):
                    a = np.asarray(action_dict[name], dtype=np.float32)
                    return a[0] if a.ndim == 3 else a   # (T, D)

                if args_cli.use_chest_cameras:
                    # EE-IK path: decode left/right eef per timestep into [pos(3), quat(4)],
                    # and decode the 20-d effector encoder chunk into a 32-d hand chunk.
                    la = _ax("left_eef")    # (T, 9) abs pos+rot6d
                    ra = _ax("right_eef")   # (T, 9)
                    eff = _ax("effector")   # (T, 20) abs encoder counts
                    T = la.shape[0]
                    eef_chunk = np.zeros((T, 14), dtype=np.float32)
                    hand_chunk32 = np.zeros((T, 32), dtype=np.float32)
                    for t in range(T):
                        l_pos, l_quat = decode_eef_pose_from_action(la[t])
                        r_pos, r_quat = decode_eef_pose_from_action(ra[t])
                        eef_chunk[t, 0:3]  = l_pos
                        eef_chunk[t, 3:7]  = l_quat
                        eef_chunk[t, 7:10] = r_pos
                        eef_chunk[t, 10:14] = r_quat
                        active_20_rad = decode_hand_from_encoder(eff[t])
                        hand_chunk32[t] = active20_to_full_hand32(active_20_rad)
                    print(f"  step {step}: eef_chunk={eef_chunk.shape} "
                          f"hand_chunk32={hand_chunk32.shape} "
                          f"eff_enc[min..max]=[{float(eff.min()):.0f}..{float(eff.max()):.0f}] "
                          f"infer={time.time() - t0:.2f}s")
                else:
                    arm_rel = _ax("arm")                    # (T, 14) relative deltas
                    # N1.7 RELATIVE: each row i is delta from state.arm[-1] (NOW).
                    arm_chunk = arm_now[None, :] + arm_rel  # (T, 14)
                    print(f"  step {step}: arm_chunk={arm_chunk.shape} "
                          f"infer={time.time() - t0:.2f}s")

            # Apply rows [chunk_start_row, chunk_start_row + use_length) of each chunk
            # so we skip the anchored row 0 (= current pose) and pick up larger
            # per-step displacement. Row indexing is bounds-checked at startup.
            row_idx = args_cli.chunk_start_row + (step % args_cli.use_length)
            if args_cli.use_chest_cameras:
                # 46-d Pink-IK action: 14 EE pose + 32 hand joints (decoded from
                # action.effector via per-joint linear encoder calibration).
                row = eef_chunk[row_idx]
                hand_row32 = hand_chunk32[row_idx]
                action_vec = np.concatenate([row, hand_row32], axis=0).astype(np.float32)
                action_t = torch.from_numpy(action_vec).to(args_cli.device).unsqueeze(0)
            else:
                row = arm_chunk[row_idx]
                target = cur_joint_pos.copy()
                target[arm_env_idx] = row                    # arm only
                action_t = torch.from_numpy(target).to(args_cli.device).unsqueeze(0)

            if args_cli.print_debug_every and step % args_cli.print_debug_every == 0:
                pol = env.unwrapped.observation_manager.compute()["policy"]
                cur_le = pol["left_eef_pos"][0].detach().cpu().numpy()
                cur_re = pol["right_eef_pos"][0].detach().cpu().numpy()
                cur_lq = pol["left_eef_quat"][0].detach().cpu().numpy()
                cur_rq = pol["right_eef_quat"][0].detach().cpu().numpy()
                if args_cli.use_chest_cameras:
                    tgt_l = row[0:3]; tgt_lq = row[3:7]
                    tgt_r = row[7:10]; tgt_rq = row[10:14]
                    arm_rel_preview = np.asarray(action_dict["arm"], dtype=np.float32)
                    arm_rel_preview = arm_rel_preview[0] if arm_rel_preview.ndim == 3 else arm_rel_preview
                    ar0 = arm_rel_preview[0]
                    print(
                        f"  [dbg s{step:04d}] "
                        f"L cur={cur_le.round(3).tolist()} → tgt={tgt_l.round(3).tolist()} "
                        f"|Δ|={float(np.linalg.norm(tgt_l - cur_le)):.4f}m  "
                        f"R cur={cur_re.round(3).tolist()} → tgt={tgt_r.round(3).tolist()} "
                        f"|Δ|={float(np.linalg.norm(tgt_r - cur_re)):.4f}m  "
                        f"arm_rel[0] |max|={float(np.max(np.abs(ar0))):.4f}rad"
                    )
                    print(
                        f"  [dbg s{step:04d}] "
                        f"L quat cur={cur_lq.round(3).tolist()} tgt={tgt_lq.round(3).tolist()}  "
                        f"R quat cur={cur_rq.round(3).tolist()} tgt={tgt_rq.round(3).tolist()}"
                    )
                    # Hand: compare current 20 active joint pos (rad) vs commanded target.
                    if hand_env_idx is not None and hand_chunk32 is not None:
                        hand_now20 = cur_joint_pos[hand_env_idx]
                        hand_tgt20 = hand_chunk32[step % hand_chunk32.shape[0]][
                            np.array([0,1,2,3,4,5,6,7,8,9, 16,17,18,19,20,21,22,23,24,25])
                        ]
                        print(
                            f"  [dbg s{step:04d}] hand_now (active 20)={np.round(hand_now20, 2).tolist()}"
                        )
                        print(
                            f"  [dbg s{step:04d}] hand_tgt (active 20)={np.round(hand_tgt20, 2).tolist()} "
                            f"|Δ| max={float(np.max(np.abs(hand_tgt20 - hand_now20))):.3f}rad"
                        )
                else:
                    print(
                        f"  [dbg s{step:04d}] arm_now={arm_now.round(4).tolist()} "
                        f"arm_target={row.round(4).tolist()}"
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
