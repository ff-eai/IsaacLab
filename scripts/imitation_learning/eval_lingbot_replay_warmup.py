"""Replay-warmup eval: replay K recorded actions, then hand off to LingBot WS policy.

Lets us test whether the policy can continue a manipulation trajectory once it's
been put in a familiar mid-task pose, separating "model never learned init pose"
from "model is just bad."

Pink IK schema (38-d action). Resets the env to ``demo_0`` initial state from
the same source HDF5 the model was trained on, replays ``--replay_steps``
recorded actions, then asks the policy for the rest.

Usage::

    isaaclab.sh -p scripts/imitation_learning/eval_lingbot_replay_warmup.py \
        --dataset_file /home/wagner/2T/wagner/dataset/issac_placn/a2_pickplace_v2_generated.hdf5 \
        --demo demo_0 --replay_steps 100 \
        --websocket_host 127.0.0.1 --websocket_port 50963 \
        --max_steps 400 --use_length 50 --debug_ik_every 20
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

import pinocchio  # noqa: F401
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay-warmup eval for LingBot-VLA.")
parser.add_argument("--task_id", type=str, default="Isaac-PickPlace-A2-Abs-v0")
parser.add_argument("--task", type=str, default="place the can in the tray")
parser.add_argument("--dataset_file", type=str, required=True)
parser.add_argument("--demo", type=str, default="demo_0")
parser.add_argument("--replay_steps", type=int, default=100,
                    help="How many recorded actions to replay before handoff.")
parser.add_argument("--websocket_host", type=str, default="127.0.0.1")
parser.add_argument("--websocket_port", type=int, default=50963)
parser.add_argument("--max_steps", type=int, default=400)
parser.add_argument("--use_length", type=int, default=50)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--lerobot_path", type=str,
                    default="/home/wagner/code/lingbot-vla")
parser.add_argument("--debug_ik_every", type=int, default=20)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import h5py
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

sys.path.insert(0, args_cli.lerobot_path)
from deploy.websocket_client_policy import WebsocketClientPolicy  # noqa: E402


MIMIC_CHILD_JOINTS = (
    "L_index_2_joint", "L_middle_2_joint", "L_pinky_2_joint", "L_ring_2_joint",
    "R_index_2_joint", "R_middle_2_joint", "R_pinky_2_joint", "R_ring_2_joint",
    "L_thumb_2_joint", "R_thumb_2_joint",
    "L_thumb_3_joint", "R_thumb_3_joint",
)
_HAND_JOINT_NAMES_IN_ACTION = (
    "L_index_1_joint", "L_middle_1_joint", "L_pinky_1_joint", "L_ring_1_joint", "L_thumb_swing_joint",
    "R_index_1_joint", "R_middle_1_joint", "R_pinky_1_joint", "R_ring_1_joint", "R_thumb_swing_joint",
    "L_index_2_joint", "L_middle_2_joint", "L_pinky_2_joint", "L_ring_2_joint", "L_thumb_1_joint",
    "R_index_2_joint", "R_middle_2_joint", "R_pinky_2_joint", "R_ring_2_joint", "R_thumb_1_joint",
    "L_thumb_2_joint", "R_thumb_2_joint", "L_thumb_3_joint", "R_thumb_3_joint",
)


def _quat_angle_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    q1 = q1 / max(1e-12, float(np.linalg.norm(q1)))
    q2 = q2 / max(1e-12, float(np.linalg.norm(q2)))
    dot = abs(float(np.dot(q1, q2)))
    dot = min(1.0, max(-1.0, dot))
    return float(np.degrees(2.0 * np.arccos(dot)))


def encode_obs_for_policy(env, task: str, mimic_state_idx) -> dict:
    obs_dict = env.unwrapped.observation_manager.compute()
    pol = obs_dict["policy"]

    def _img_hwc(t: torch.Tensor) -> np.ndarray:
        return t[0].detach().to(torch.uint8).cpu().numpy()

    head = _img_hwc(pol["head_camera_rgb"])
    cleft = _img_hwc(pol["chest_left_camera_rgb"])
    cright = _img_hwc(pol["chest_right_camera_rgb"])
    state_53 = pol["robot_joint_pos"][0].detach().cpu().numpy().astype(np.float32)
    state_53[mimic_state_idx] = 0.0
    return {
        "observation.images.cam_high": head,
        "observation.images.cam_left_wrist": cleft,
        "observation.images.cam_right_wrist": cright,
        "observation.state": state_53,
        "task": task,
    }


def main():
    cfg = parse_env_cfg(args_cli.task_id, device=args_cli.device, num_envs=args_cli.num_envs)
    cfg.teleop_devices.devices = {}
    if hasattr(cfg, "recorders"):
        cfg.recorders = None
    cfg.actions.upper_body_ik.controller.show_ik_warnings = True
    cfg.actions.upper_body_ik.controller.fail_on_joint_limit_violation = False

    print(f"[eval] making env {args_cli.task_id}")
    env = gym.make(args_cli.task_id, cfg=cfg).unwrapped
    env.seed = args_cli.seed
    device = env.device

    joint_names = list(env.scene["robot"].data.joint_names)
    body_names = list(env.scene["robot"].data.body_names)
    left_eef_idx = body_names.index("left_arm_link07")
    right_eef_idx = body_names.index("right_arm_link07")
    mimic_state_idx = np.asarray([joint_names.index(n) for n in MIMIC_CHILD_JOINTS], dtype=np.int64)
    mimic_hand_idx = np.asarray([_HAND_JOINT_NAMES_IN_ACTION.index(n) for n in MIMIC_CHILD_JOINTS], dtype=np.int64)
    print(f"[eval] action_dim={env.action_manager.total_action_dim} replay_steps={args_cli.replay_steps}")

    print(f"[eval] loading {args_cli.demo} from {args_cli.dataset_file}")
    with h5py.File(args_cli.dataset_file, "r") as f:
        demo = f[f"data/{args_cli.demo}"]
        recorded_actions = demo["actions"][:].astype(np.float32)
        init = demo["initial_state"]
        init_robot_q  = init["articulation/robot/joint_position"][0]
        init_robot_qv = init["articulation/robot/joint_velocity"][0]
        init_robot_pos = init["articulation/robot/root_pose"][0, :3]
        init_robot_rot = init["articulation/robot/root_pose"][0, 3:7]
        init_obj_pose  = init["rigid_object/object/root_pose"][0]
        init_tray_pose = init["rigid_object/tray/root_pose"][0]
    print(f"[eval] recorded_actions shape={recorded_actions.shape}")

    env_origin = env.scene.env_origins[0].cpu().numpy()
    env.reset()
    robot = env.scene["robot"]
    q = torch.from_numpy(init_robot_q).to(device, dtype=torch.float32).unsqueeze(0)
    qv = torch.from_numpy(init_robot_qv).to(device, dtype=torch.float32).unsqueeze(0)
    robot.write_joint_state_to_sim(q, qv)
    root = np.concatenate([init_robot_pos, init_robot_rot, np.zeros(6, dtype=np.float32)]).astype(np.float32)
    robot.write_root_state_to_sim(torch.from_numpy(root).to(device).unsqueeze(0))
    for name, pose in (("object", init_obj_pose), ("tray", init_tray_pose)):
        if name in env.scene.rigid_objects:
            r = np.concatenate([pose, np.zeros(6, dtype=np.float32)]).astype(np.float32)
            env.scene.rigid_objects[name].write_root_state_to_sim(
                torch.from_numpy(r).to(device).unsqueeze(0))

    print(f"[eval] connecting ws://{args_cli.websocket_host}:{args_cli.websocket_port}")
    policy = WebsocketClientPolicy(host=args_cli.websocket_host, port=args_cli.websocket_port)
    try:
        policy.reset("isaac_a2")
    except Exception as e:
        print(f"  policy.reset warning: {e}")

    n_replay = min(args_cli.replay_steps, recorded_actions.shape[0])
    chunk = None
    success = False
    for step in range(args_cli.max_steps):
        if step < n_replay:
            row = recorded_actions[step].copy()
            mode = "REPLAY"
        else:
            policy_step = step - n_replay
            if policy_step % max(1, args_cli.use_length) == 0:
                payload = encode_obs_for_policy(env, args_cli.task, mimic_state_idx)
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
            row = chunk[policy_step % chunk.shape[0]].copy()
            row[14 + mimic_hand_idx] = 0.0
            mode = "POLICY"

        if row.shape[0] != 38:
            row = row[:38]

        if args_cli.debug_ik_every and step % args_cli.debug_ik_every == 0:
            bs_pre = robot.data.body_state_w[0]
            ach_l_pre = bs_pre[left_eef_idx, :3].detach().cpu().numpy() - env_origin
            ach_r_pre = bs_pre[right_eef_idx, :3].detach().cpu().numpy() - env_origin
            tgt_l = row[0:3]; tgt_r = row[7:10]
            print(
                f"  step {step:3d} [{mode}] L tgt={tgt_l.round(3).tolist()} ach={ach_l_pre.round(3).tolist()} "
                f"|Δ|={float(np.linalg.norm(ach_l_pre - tgt_l)):.3f} | "
                f"R tgt={tgt_r.round(3).tolist()} ach={ach_r_pre.round(3).tolist()} "
                f"|Δ|={float(np.linalg.norm(ach_r_pre - tgt_r)):.3f}"
            )

        action_t = torch.from_numpy(row).to(device).unsqueeze(0)
        obs, rew, terminated, truncated, info = env.step(action_t)
        term = bool(terminated[0]) if hasattr(terminated, "__getitem__") else bool(terminated)
        trunc = bool(truncated[0]) if hasattr(truncated, "__getitem__") else bool(truncated)
        if term or trunc:
            success_flag = False
            done_mgr = getattr(env, "termination_manager", None)
            if done_mgr is not None and "success" in done_mgr.active_terms:
                s_buf = done_mgr.get_term("success")
                success_flag = bool(s_buf[0])
            print(f"  step {step}: terminated={term} truncated={trunc} success={success_flag} (mode={mode})")
            success = success_flag
            break
    print(f"[eval] DONE — success={success}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
