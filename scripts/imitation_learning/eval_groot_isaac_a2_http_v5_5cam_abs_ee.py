"""Closed-loop eval for GR00T v5 (5cam + absolute arm/effector + relative EE).

Pairs with HTTP server on 172.18.1.26:18008 and checkpoint-30000.

Default env: Isaac-PickPlace-A2OmniHand-Abs-v0 (10-DOF OmniHand per hand, 20-d effector).
Use Isaac-PickPlace-A2-Abs-v0 for stock s6_hand (12 finger joints; effector sent as zeros).
"""

from __future__ import annotations

import argparse
import functools
import http.client
import os
import pickle
import sys
import time

print = functools.partial(print, flush=True)  # noqa: A001
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import pinocchio  # noqa: F401

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="GR00T v5 5cam HTTP eval (absolute arm).")
parser.add_argument("--task_id", type=str, default="Isaac-PickPlace-A2OmniHand-Abs-v0")
parser.add_argument("--task", type=str, default="place the can in the tray")
parser.add_argument("--groot_host", type=str, default="172.18.1.26")
parser.add_argument("--groot_port", type=int, default=18008)
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument("--max_steps", type=int, default=2000)
parser.add_argument("--use_length", type=int, default=16)
parser.add_argument("--chunk_start_row", type=int, default=0)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--action_mode",
    choices=("joint", "eef_ik"),
    default="joint",
    help="joint: action.arm + action.effector as absolute joint targets. "
         "eef_ik: Pink IK from absolute left_eef/right_eef + effector hands.",
)
parser.add_argument("--print_arms_every", type=int, default=50)
parser.add_argument(
    "--bottle_side",
    type=str,
    default="right",
    choices=("left", "right", "random"),
    help="Bottle spawn side relative to tray (+x=right). Default right for right-arm reach.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.num_envs != 1:
    raise SystemExit("Only num_envs=1 is supported.")
if args_cli.chunk_start_row + args_cli.use_length > 16:
    raise SystemExit("chunk_start_row + use_length must be <= 16")

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
from isaaclab_tasks.manager_based.manipulation.pick_place.pickplace_a2_env_cfg import (
    enable_pickplace_a2_5cameras,
)
from isaaclab_tasks.manager_based.manipulation.pick_place.pickplace_a2_omnihand_env_cfg import (
    enable_pickplace_omnihand_5cameras,
)
from isaaclab_tasks.utils import parse_env_cfg

PINK_IK_ACTION_DIM_S6 = 26
PINK_IK_ACTION_DIM_OMNI = 46

ARM_JOINT_NAMES = [
    "idx13_left_arm_joint1", "idx14_left_arm_joint2", "idx15_left_arm_joint3",
    "idx16_left_arm_joint4", "idx17_left_arm_joint5", "idx18_left_arm_joint6",
    "idx19_left_arm_joint7",
    "idx20_right_arm_joint1", "idx21_right_arm_joint2", "idx22_right_arm_joint3",
    "idx23_right_arm_joint4", "idx24_right_arm_joint5", "idx25_right_arm_joint6",
    "idx26_right_arm_joint7",
]
S6_HAND_JOINT_NAMES = [
    "L_index_1_joint", "L_middle_1_joint", "L_pinky_1_joint", "L_ring_1_joint",
    "L_thumb_swing_joint", "L_thumb_1_joint",
    "R_index_1_joint", "R_middle_1_joint", "R_pinky_1_joint", "R_ring_1_joint",
    "R_thumb_swing_joint", "R_thumb_1_joint",
]
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
EFFECTOR_DIM = 20
EFFECTOR_FULL_RANGE = 4000.0
_HAND_LIMITS_LOWER = np.concatenate([
    np.array([-0.0297, -1.6424, 0.0, -0.1641, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], np.float32),
    np.array([-0.0297, -1.6424, 0.0, -0.1641, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], np.float32),
])
_HAND_LIMITS_UPPER = np.concatenate([
    np.array([1.1214, 0.0454, 0.8416, 0.0, 1.4835, 1.4835, 0.1693, 1.4835, 0.1850, 1.4835], np.float32),
    np.array([1.1214, 0.0454, 0.8416, 0.0, 1.4835, 1.4835, 0.1693, 1.4835, 0.1850, 1.4835], np.float32),
])
_HAND_LIMITS_RANGE = _HAND_LIMITS_UPPER - _HAND_LIMITS_LOWER
_FULL_TO_ACTIVE_IDX = np.array([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    2, 2, 4, 5, 7, 9,
    10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    12, 12, 14, 15, 17, 19,
], dtype=np.int64)
_FULL_MIMIC_MULT = np.array([
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.33, 1.30, 1.097, 1.097, 1.097, 1.097,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.33, 1.30, 1.097, 1.097, 1.097, 1.097,
], dtype=np.float32)


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


def _resize_rgb(img: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def _quat_wxyz_to_rot6d(q: np.ndarray) -> np.ndarray:
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
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
        w, x, y, z = 0.25 / s, (m21 - m12) * s, (m02 - m20) * s, (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        w, x, y, z = (m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        w, x, y, z = (m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        w, x, y, z = (m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float32)
    return q / max(np.linalg.norm(q), 1e-12)


def decode_eef_pose_from_action(eef9: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos = eef9[0:3].astype(np.float32)
    quat = _matrix_to_quat_wxyz(_rot6d_to_matrix(eef9[3:9]))
    return pos, quat


def encode_hand_to_encoder(rad_20: np.ndarray) -> np.ndarray:
    n = (rad_20.astype(np.float32) - _HAND_LIMITS_LOWER) / _HAND_LIMITS_RANGE
    return np.clip(n, 0.0, 1.0) * EFFECTOR_FULL_RANGE


def decode_hand_from_encoder(enc_20: np.ndarray) -> np.ndarray:
    n = np.clip(enc_20.astype(np.float32) / EFFECTOR_FULL_RANGE, 0.0, 1.0)
    return n * _HAND_LIMITS_RANGE + _HAND_LIMITS_LOWER


def active20_to_full_hand32(active_20: np.ndarray) -> np.ndarray:
    return (active_20.astype(np.float32)[_FULL_TO_ACTIVE_IDX] * _FULL_MIMIC_MULT)


def encode_obs_v5(env, arm_pos_14: np.ndarray, task: str, hand_pos_20: np.ndarray | None) -> dict:
    pol = env.unwrapped.observation_manager.compute()["policy"]
    head = _resize_rgb(_img_hwc(pol["head_camera_rgb"]), 640, 360)
    chest_l = _img_hwc(pol["chest_left_camera_rgb"])
    chest_r = _img_hwc(pol["chest_right_camera_rgb"])
    hand_l = _img_hwc(pol["left_wrist_camera_rgb"])
    hand_r = _img_hwc(pol["right_wrist_camera_rgb"])
    le_pos = pol["left_eef_pos"][0].detach().cpu().numpy().astype(np.float32)
    le_quat = pol["left_eef_quat"][0].detach().cpu().numpy().astype(np.float32)
    re_pos = pol["right_eef_pos"][0].detach().cpu().numpy().astype(np.float32)
    re_quat = pol["right_eef_quat"][0].detach().cpu().numpy().astype(np.float32)
    le = np.concatenate([le_pos, _quat_wxyz_to_rot6d(le_quat)], axis=0)
    re = np.concatenate([re_pos, _quat_wxyz_to_rot6d(re_quat)], axis=0)
    if hand_pos_20 is not None:
        eff_in = encode_hand_to_encoder(hand_pos_20)
    else:
        eff_in = np.zeros(EFFECTOR_DIM, dtype=np.float32)
    return {
        "video": {
            "head_front_color": head[None, None, ...].astype(np.uint8),
            "chest_left": chest_l[None, None, ...].astype(np.uint8),
            "chest_right": chest_r[None, None, ...].astype(np.uint8),
            "hand_left": hand_l[None, None, ...].astype(np.uint8),
            "hand_right": hand_r[None, None, ...].astype(np.uint8),
        },
        "state": {
            "effector": eff_in.astype(np.float32)[None, None, :],
            "arm": arm_pos_14.astype(np.float32)[None, None, :],
            "left_eef": le.astype(np.float32)[None, None, :],
            "right_eef": re.astype(np.float32)[None, None, :],
        },
        "language": {
            "annotation.human.task_description": [[task]],
        },
    }


def post_act(host: str, port: int, payload: dict, timeout: float = 60.0) -> dict:
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
        return pickle.loads(resp.read())
    finally:
        conn.close()


def main():
    omnihand = "OmniHand" in args_cli.task_id
    cfg = parse_env_cfg(args_cli.task_id, device=args_cli.device, num_envs=args_cli.num_envs)
    if omnihand:
        enable_pickplace_omnihand_5cameras(cfg)
        print("[eval] OmniHand env: 5 cameras + 20-d effector encoding")
    else:
        enable_pickplace_a2_5cameras(cfg)
        print("[eval] s6_hand env: 5 cameras; effector zeros in obs")

    if args_cli.action_mode == "joint":
        cfg.actions = _JointActOnlyCfg()
        print("[eval] action_mode=joint (v5 arm + effector absolute -> joints)")
    else:
        print("[eval] action_mode=eef_ik (absolute EEF + effector -> Pink IK 46-d)")

    cfg.teleop_devices.devices = {}
    if hasattr(cfg, "recorders"):
        cfg.recorders = None
    cfg.episode_length_s = max(args_cli.max_steps * 0.05 + 5.0, getattr(cfg, "episode_length_s", 0.0))

    if args_cli.bottle_side != "random":
        cfg.events.reset_object.params["force_side"] = args_cli.bottle_side
        print(f"[eval] bottle spawn: fixed {args_cli.bottle_side} of tray")

    print(f"[eval] making env {args_cli.task_id} on {args_cli.device}")
    env = gym.make(args_cli.task_id, cfg=cfg).unwrapped
    env.seed = args_cli.seed
    if omnihand:
        robot_z = float(env.scene["robot"].data.root_pos_w[0, 2])
        tray_z = float(env.scene["tray"].data.root_pos_w[0, 2])
        print(f"[eval] scene heights: robot_base_z={robot_z:.2f} tray_z={tray_z:.2f} (tabletop~0.90)")

    env_joint_names = list(env.scene["robot"].data.joint_names)
    env_name_to_idx = {n: i for i, n in enumerate(env_joint_names)}
    arm_env_idx = np.asarray([env_name_to_idx[n] for n in ARM_JOINT_NAMES], dtype=np.int64)

    hand_active_idx = hand32_env_idx = s6_hand_idx = None
    if omnihand:
        hand_active_idx = np.asarray([env_name_to_idx[n] for n in ACTIVE_HAND_JOINT_NAMES], dtype=np.int64)
        hand32_env_idx = np.asarray([env_name_to_idx[n] for n in OMNIHAND_HAND_JOINT_NAMES], dtype=np.int64)
    else:
        s6_hand_idx = np.asarray([env_name_to_idx[n] for n in S6_HAND_JOINT_NAMES], dtype=np.int64)

    act_dim = env.action_manager.total_action_dim
    print(f"[eval] joints={len(env_joint_names)} action_dim={act_dim}")
    expected_ik = PINK_IK_ACTION_DIM_OMNI if omnihand else PINK_IK_ACTION_DIM_S6
    if args_cli.action_mode == "eef_ik" and act_dim != expected_ik:
        print(f"  WARN: expected Pink-IK dim {expected_ik}; got {act_dim}")

    print(f"[eval] GR00T v5 HTTP {args_cli.groot_host}:{args_cli.groot_port}/act")

    successes, episode_lengths = 0, []
    for ep in range(args_cli.episodes):
        env.reset()
        arm_chunk = eff_chunk = eef_chunk = hand_chunk32 = None
        success = False
        for step in range(args_cli.max_steps):
            cur_joint_pos = env.scene["robot"].data.joint_pos[0].detach().cpu().numpy().astype(np.float32)
            arm_now = cur_joint_pos[arm_env_idx]
            hand_now20 = cur_joint_pos[hand_active_idx] if hand_active_idx is not None else None
            payload = encode_obs_v5(env, arm_now, args_cli.task, hand_pos_20=hand_now20)

            if step % max(1, args_cli.use_length) == 0:
                t0 = time.time()
                try:
                    action_dict = post_act(args_cli.groot_host, args_cli.groot_port, payload)
                except Exception as e:
                    print(f"  step {step}: HTTP /act failed: {e!r}")
                    break

                def _ax(name):
                    a = np.asarray(action_dict[name], dtype=np.float32)
                    return a[0] if a.ndim == 3 else a

                arm_chunk = _ax("arm")
                eff_chunk = _ax("effector")
                la = _ax("left_eef")
                ra = _ax("right_eef")
                T = arm_chunk.shape[0]
                if args_cli.action_mode == "eef_ik":
                    eef_chunk = np.zeros((T, 14), dtype=np.float32)
                    hand_chunk32 = np.zeros((T, 32), dtype=np.float32)
                    for t in range(T):
                        l_pos, l_quat = decode_eef_pose_from_action(la[t])
                        r_pos, r_quat = decode_eef_pose_from_action(ra[t])
                        eef_chunk[t, 0:3], eef_chunk[t, 3:7] = l_pos, l_quat
                        eef_chunk[t, 7:10], eef_chunk[t, 10:14] = r_pos, r_quat
                        hand_chunk32[t] = active20_to_full_hand32(decode_hand_from_encoder(eff_chunk[t]))
                print(
                    f"  step {step}: arm {arm_chunk.shape} effector {eff_chunk.shape} "
                    f"infer={time.time() - t0:.2f}s"
                )

            row_idx = args_cli.chunk_start_row + (step % args_cli.use_length)

            if args_cli.action_mode == "joint":
                target = cur_joint_pos.copy()
                target[arm_env_idx] = arm_chunk[row_idx]
                if omnihand:
                    active_rad = decode_hand_from_encoder(eff_chunk[row_idx])
                    target[hand_active_idx] = active_rad
                    target[hand32_env_idx] = active20_to_full_hand32(active_rad)
                elif s6_hand_idx is not None:
                    target[s6_hand_idx] = cur_joint_pos[s6_hand_idx]
                action_t = torch.from_numpy(target).to(args_cli.device).unsqueeze(0)
                if args_cli.print_arms_every > 0 and step % args_cli.print_arms_every == 0:
                    j = int(arm_env_idx[7])
                    print(
                        f"  step {step} right_arm_j1: cur={cur_joint_pos[j]:+.4f} "
                        f"tgt={target[j]:+.4f} delta={target[j] - cur_joint_pos[j]:+.4f}"
                    )
            else:
                row_eef = eef_chunk[row_idx]
                row_hand = hand_chunk32[row_idx]
                action_vec = np.concatenate([row_eef, row_hand], axis=0).astype(np.float32)
                action_t = torch.from_numpy(action_vec).to(args_cli.device).unsqueeze(0)

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
        if success:
            successes += 1
        print(f"[eval] episode {ep}: success={success} len={episode_lengths[-1]}")

    print(f"[eval] DONE — {successes}/{args_cli.episodes} success")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
