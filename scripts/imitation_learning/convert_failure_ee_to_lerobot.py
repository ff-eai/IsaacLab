"""Convert labeled failure HDF5 (with EE pose) -> LeRobot v2 dataset.

State = obs/robot_joint_pos (N) + left_eef_pos(3) + left_eef_quat(4) +
        right_eef_pos(3) + right_eef_quat(4) = (N+14) dims

Action = data/actions (M-dim, detected from HDF5; typically 26 for Mimic env)

Images are included only if the source HDF5 has camera fields
(obs/head_camera_rgb, obs/chest_left_camera_rgb, obs/chest_right_camera_rgb,
 obs/head_camera_depth). Many synthetic-generation HDF5s drop camera frames.

Run:
  python convert_failure_ee_to_lerobot.py \\
      --input_file /path/to/a2_pickplace_failed_labeled.hdf5 \\
      --joint_names_json /path/to/a2_joint_names.json \\
      --repo_id a2_pickplace_failed_3500_ee \\
      --output_dir /path/to/lerobot_cache

Requires: lerobot, h5py, cv2.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import tqdm


def resize_chw_480x640(rgb_hwc: np.ndarray) -> np.ndarray:
    """Resize a HxWxC uint8 RGB frame to 480x640, then HWC -> CHW."""
    if rgb_hwc.shape[:2] != (480, 640):
        rgb_hwc = cv2.resize(rgb_hwc, (640, 480), interpolation=cv2.INTER_AREA)
    return np.transpose(rgb_hwc, (2, 0, 1))


def depth_to_chw_480x640(depth: np.ndarray, max_m: float = 4.0) -> np.ndarray:
    """Convert a HxWx1 (or HxW) depth-in-metres frame -> 3xHxW uint8 image."""
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.shape[:2] != (480, 640):
        depth = cv2.resize(depth, (640, 480), interpolation=cv2.INTER_NEAREST)
    depth = np.clip(depth, 0.0, max_m)
    depth_u8 = (depth / max_m * 255.0).astype(np.uint8)
    return np.stack([depth_u8, depth_u8, depth_u8], axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--joint_names_json", type=str, required=True)
    parser.add_argument("--repo_id", type=str, default="a2_pickplace_failed_3500_ee")
    parser.add_argument("--task", type=str, default="place the can in the tray")
    parser.add_argument(
        "--lerobot_home",
        type=str,
        default=None,
        help="Override HF_LEROBOT_HOME.",
    )
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    if args.lerobot_home:
        os.environ["HF_LEROBOT_HOME"] = args.lerobot_home

    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

    with open(args.joint_names_json) as f:
        joint_names = json.load(f)["joint_names"]
    joint_dim = len(joint_names)
    print(f"[convert-ee] {joint_dim} env joints in joint_names JSON")

    src = Path(args.input_file)
    if not src.exists():
        raise FileNotFoundError(src)

    # Peek first episode to learn actual joint dim and action dim.
    with h5py.File(src, "r") as f:
        if "data" in f:
            grp = f["data"]
        elif "failed" in f:
            grp = f["failed"]
        else:
            raise ValueError(f"Cannot find 'data' or 'failed' group in {src}")
        first_key = sorted(grp.keys(), key=lambda k: int(k.split("_")[-1]))[0]
        action_dim = int(grp[first_key]["actions"].shape[1])
        actual_joint_dim = int(grp[first_key]["obs"]["robot_joint_pos"].shape[1])
        print(f"[convert-ee] action dim from hdf5: {action_dim}")
        print(f"[convert-ee] actual robot_joint_pos dim: {actual_joint_dim}")

    # State dim is driven by the actual joint data, not the joint_names JSON.
    state_dim = actual_joint_dim + 14
    if joint_dim != actual_joint_dim:
        print(f"[convert-ee] WARNING: joint_names has {joint_dim} names but data has {actual_joint_dim} dims")
        print(f"[convert-ee] Using {actual_joint_dim} EE-augmented state (names truncated to match)")

    state_names = joint_names[:actual_joint_dim] + [
        "left_eef_pos_x", "left_eef_pos_y", "left_eef_pos_z",
        "left_eef_quat_x", "left_eef_quat_y", "left_eef_quat_z", "left_eef_quat_w",
        "right_eef_pos_x", "right_eef_pos_y", "right_eef_pos_z",
        "right_eef_quat_x", "right_eef_quat_y", "right_eef_quat_z", "right_eef_quat_w",
    ]

    if (HF_LEROBOT_HOME / args.repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / args.repo_id)

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": [state_names],
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": [[f"a{i}" for i in range(action_dim)]],
        },
    }

    # Detect cameras from the first episode.
    with h5py.File(src, "r") as f:
        if "data" in f:
            check_grp = f["data"]
        elif "failed" in f:
            check_grp = f["failed"]
        else:
            raise ValueError(f"No 'data' or 'failed' group in {src}")
        check_ep = check_grp[first_key]
        obs_keys = list(check_ep["obs"].keys())
        has_cameras = all(k in obs_keys for k in
                          ("head_camera_rgb", "chest_left_camera_rgb", "chest_right_camera_rgb"))
        has_depth = "head_camera_depth" in obs_keys

    if has_cameras:
        for cam in ("cam_high", "cam_chest_left", "cam_chest_right"):
            features[f"observation.images.{cam}"] = {
                "dtype": "image",
                "shape": (3, 480, 640),
                "names": ["channels", "height", "width"],
            }
    else:
        print("[convert-ee] WARNING: No camera images found in source — state-only dataset")
    if has_depth:
        features["observation.images.cam_high_depth"] = {
            "dtype": "image",
            "shape": (3, 480, 640),
            "names": ["channels", "height", "width"],
        }

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        robot_type="a2_humanoid",
        features=features,
        use_videos=False,
        image_writer_processes=10,
        image_writer_threads=5,
    )

    with h5py.File(src, "r") as f:
        if "data" in f:
            ep_grp = f["data"]
        elif "failed" in f:
            ep_grp = f["failed"]
        else:
            raise ValueError(f"No 'data' or 'failed' group in {src}")

        demo_keys = sorted(ep_grp.keys(), key=lambda k: int(k.split("_")[-1]))
        print(f"[convert-ee] {len(demo_keys)} episodes -> {HF_LEROBOT_HOME / args.repo_id}")
        skipped = 0
        for k in tqdm.tqdm(demo_keys):
            ep = ep_grp[k]
            joint_pos = ep["obs"]["robot_joint_pos"][:]

            # Check all EE components.
            if not all(key in ep["obs"] for key in ("left_eef_pos", "left_eef_quat",
                                                      "right_eef_pos", "right_eef_quat")):
                skipped += 1
                continue
            left_eef_pos = ep["obs"]["left_eef_pos"][:]
            left_eef_quat = ep["obs"]["left_eef_quat"][:]
            right_eef_pos = ep["obs"]["right_eef_pos"][:]
            right_eef_quat = ep["obs"]["right_eef_quat"][:]

            # State = joint_pos + left EE + right EE
            state = np.concatenate([
                joint_pos,
                left_eef_pos, left_eef_quat,
                right_eef_pos, right_eef_quat,
            ], axis=-1).astype(np.float32)

            action = ep["actions"][:].astype(np.float32)

            T = state.shape[0]
            assert action.shape[0] == T, (
                f"length mismatch in {k}: state={T} action={action.shape[0]}"
            )

            for i in range(T):
                frame = {
                    "observation.state": torch.from_numpy(state[i]),
                    "action": torch.from_numpy(action[i]),
                }
                if has_cameras:
                    frame["observation.images.cam_high"] = resize_chw_480x640(
                        ep["obs"]["head_camera_rgb"][i])
                    frame["observation.images.cam_chest_left"] = resize_chw_480x640(
                        ep["obs"]["chest_left_camera_rgb"][i])
                    frame["observation.images.cam_chest_right"] = resize_chw_480x640(
                        ep["obs"]["chest_right_camera_rgb"][i])
                if has_depth:
                    frame["observation.images.cam_high_depth"] = depth_to_chw_480x640(
                        ep["obs"]["head_camera_depth"][i])
                dataset.add_frame(frame, task=args.task)
            dataset.save_episode()

        if skipped:
            print(f"[convert-ee] skipped {skipped} episodes (missing EE data)")
    print("[convert-ee] done.")


if __name__ == "__main__":
    main()
