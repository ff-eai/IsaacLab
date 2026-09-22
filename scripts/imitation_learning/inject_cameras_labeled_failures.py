"""Replay labeled failure episodes in sim and write 3 RGB cameras into HDF5.

Uses recorded ``states/*`` per frame (kinematic replay) so rendered cameras match
the logged trajectory. Output is a copy of the input HDF5 with:
  obs/head_camera_rgb, obs/chest_left_camera_rgb, obs/chest_right_camera_rgb

Usage::

    isaaclab.sh -p scripts/imitation_learning/inject_cameras_labeled_failures.py \\
        --headless --enable_cameras --enable_pinocchio \\
        --input_file /path/to/a2_pickplace_failed_balanced_labeled.hdf5 \\
        --output_file /path/to/a2_pickplace_failed_balanced_labeled_cam.hdf5 \\
        --max_per_filter 100 --seed 42
"""

from __future__ import annotations

import argparse
import functools
import os
import shutil
import sys
from pathlib import Path

print = functools.partial(print, flush=True)  # noqa: A001
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import pinocchio  # noqa: F401
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Inject 3 RGB cameras into labeled failure HDF5.")
parser.add_argument("--input_file", type=str, required=True)
parser.add_argument("--output_file", type=str, required=True)
parser.add_argument("--task_id", type=str, default="Isaac-PickPlace-A2-Mimic-v0")
parser.add_argument("--max_per_filter", type=int, default=100)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--start_episode", type=int, default=0, help="Skip first N selected episodes.")
parser.add_argument("--max_episodes", type=int, default=None, help="Cap episodes processed this run.")
parser.add_argument("--overwrite", action="store_true", help="Replace output_file if it exists.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import h5py
import numpy as np
import torch
import tqdm

import isaaclab_tasks  # noqa: F401
import isaaclab_mimic.envs.pinocchio_envs  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from convert_isaac_a2_labeled_to_lerobot import sample_disjoint_episodes  # noqa: E402


def _img_hwc_uint8(t: torch.Tensor) -> np.ndarray:
    arr = t[0].detach().cpu().numpy()
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _apply_frame_state(env, states_grp: h5py.Group, idx: int, device: torch.device) -> None:
    robot = env.scene["robot"]
    q = torch.from_numpy(states_grp["articulation/robot/joint_position"][idx]).to(device, dtype=torch.float32).unsqueeze(0)
    qv = torch.from_numpy(states_grp["articulation/robot/joint_velocity"][idx]).to(device, dtype=torch.float32).unsqueeze(0)
    robot.write_joint_state_to_sim(q, qv)
    root = np.concatenate(
        [
            states_grp["articulation/robot/root_pose"][idx],
            states_grp["articulation/robot/root_velocity"][idx],
        ]
    ).astype(np.float32)
    robot.write_root_state_to_sim(torch.from_numpy(root).to(device, dtype=torch.float32).unsqueeze(0))
    for obj_name in ("object", "tray"):
        if obj_name not in env.scene.rigid_objects:
            continue
        obj = env.scene.rigid_objects[obj_name]
        pose = np.concatenate(
            [
                states_grp[f"rigid_object/{obj_name}/root_pose"][idx],
                states_grp[f"rigid_object/{obj_name}/root_velocity"][idx],
            ]
        ).astype(np.float32)
        obj.write_root_state_to_sim(torch.from_numpy(pose).to(device, dtype=torch.float32).unsqueeze(0))
    env.scene.write_data_to_sim()
    env.sim.forward()


def _apply_initial_state(env, init_grp: h5py.Group, device: torch.device) -> None:
    robot = env.scene["robot"]
    q = torch.from_numpy(init_grp["articulation/robot/joint_position"][0]).to(device, dtype=torch.float32).unsqueeze(0)
    qv = torch.from_numpy(init_grp["articulation/robot/joint_velocity"][0]).to(device, dtype=torch.float32).unsqueeze(0)
    robot.write_joint_state_to_sim(q, qv)
    root = np.concatenate(
        [
            init_grp["articulation/robot/root_pose"][0],
            init_grp["articulation/robot/root_velocity"][0],
        ]
    ).astype(np.float32)
    robot.write_root_state_to_sim(torch.from_numpy(root).to(device, dtype=torch.float32).unsqueeze(0))
    for obj_name in ("object", "tray"):
        if obj_name not in env.scene.rigid_objects:
            continue
        obj = env.scene.rigid_objects[obj_name]
        pose = np.concatenate(
            [
                init_grp[f"rigid_object/{obj_name}/root_pose"][0],
                init_grp[f"rigid_object/{obj_name}/root_velocity"][0],
            ]
        ).astype(np.float32)
        obj.write_root_state_to_sim(torch.from_numpy(pose).to(device, dtype=torch.float32).unsqueeze(0))
    env.scene.write_data_to_sim()
    env.sim.forward()


def main():
    src = Path(args_cli.input_file)
    dst = Path(args_cli.output_file)
    if dst.exists():
        if args_cli.overwrite:
            dst.unlink()
        else:
            raise FileExistsError(f"{dst} exists; pass --overwrite to replace")

    specs = [
        ("closed_without_lift", None, "closed_without_lift"),
        ("no_hand_engagement", None, "no_hand_engagement"),
        ("placed_outside_tray", "placed_outside_tray", None),
    ]
    with h5py.File(src, "r") as h5f:
        episodes = sample_disjoint_episodes(h5f, specs, args_cli.max_per_filter, args_cli.seed)
    if args_cli.start_episode:
        episodes = episodes[args_cli.start_episode :]
    if args_cli.max_episodes is not None:
        episodes = episodes[: args_cli.max_episodes]

    shutil.copy2(src, dst)
    print(f"[inject-cam] copied base -> {dst}")
    print(f"[inject-cam] will render cameras for {len(episodes)} episodes")

    cfg = parse_env_cfg(args_cli.task_id, device=args_cli.device, num_envs=args_cli.num_envs)
    cfg.teleop_devices.devices = {}
    if hasattr(cfg, "recorders"):
        cfg.recorders = None

    env = gym.make(args_cli.task_id, cfg=cfg).unwrapped
    device = env.device
    obs_mgr = env.observation_manager

    with h5py.File(dst, "a") as h5f:
        for key, _ft, _hm in tqdm.tqdm(episodes, desc="episodes"):
            ep_in = h5f[f"failed/{key}"]
            if "states" not in ep_in:
                raise KeyError(f"failed/{key} missing states/ — cannot kinematic-replay cameras")
            T = ep_in["states/articulation/robot/joint_position"].shape[0]

            env.reset()
            _apply_initial_state(env, ep_in["initial_state"], device)
            env.scene.update(dt=env.physics_dt)

            head_frames = []
            chest_l_frames = []
            chest_r_frames = []

            for i in range(T):
                _apply_frame_state(env, ep_in["states"], i, device)
                env.scene.update(dt=env.physics_dt)
                pol = obs_mgr.compute()["policy"]
                head_frames.append(_img_hwc_uint8(pol["head_camera_rgb"]))
                chest_l_frames.append(_img_hwc_uint8(pol["chest_left_camera_rgb"]))
                chest_r_frames.append(_img_hwc_uint8(pol["chest_right_camera_rgb"]))

            ep_obs = ep_in.require_group("obs")
            for cam_key, frames in (
                ("head_camera_rgb", head_frames),
                ("chest_left_camera_rgb", chest_l_frames),
                ("chest_right_camera_rgb", chest_r_frames),
            ):
                if cam_key in ep_obs:
                    del ep_obs[cam_key]
                ep_obs.create_dataset(cam_key, data=np.stack(frames, axis=0), compression="gzip", compression_opts=4)

            print(f"[inject-cam] {key}: T={T} cameras written")

    env.close()
    print(f"[inject-cam] done -> {dst}")


if __name__ == "__main__":
    main()
    simulation_app.close()
