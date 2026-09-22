"""Convert IsaacLab Mimic annotated HDF5 -> LeRobot v2 dataset (67-dim EE state, 38-dim action, cameras).

Reads from annotated demos (produced by `annotate_demos.py` or `capture_annotated.py`)
which have full sensor data.

State: robot_joint_pos(53) + left_eef_pos(3) + left_eef_quat(4) +
       right_eef_pos(3) + right_eef_quat(4) = 67 dims

Action: actions (38-dim Pink IK: left IK 7 + right IK 7 + hand 24)

Images: head_camera_rgb(720x1280) -> 480x640, chest_left/right(480x640),
        head_camera_depth(720x1280) -> 480x640 (uint8, 3-ch)

Run:
  python convert_annotated_ee_to_lerobot.py \\
      --input_file /path/to/a2_pickplace_annotated.hdf5 \\
      --repo_id a2_pickplace_annotated_ee \\
      --lerobot_home /path/to/lerobot_cache

Requires: lerobot, h5py, cv2 (use lingbot container).
"""

import argparse
import os
import shutil
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import tqdm


def resize_chw_480x640(rgb_hwc: np.ndarray) -> np.ndarray:
    """Resize HxWxC uint8 RGB frame to 480x640 CHW."""
    if rgb_hwc.shape[:2] != (480, 640):
        rgb_hwc = cv2.resize(rgb_hwc, (640, 480), interpolation=cv2.INTER_AREA)
    return np.transpose(rgb_hwc, (2, 0, 1))


def depth_to_chw_480x640(depth_hwc: np.ndarray, max_m: float = 4.0) -> np.ndarray:
    """Resize HxWx1 (or HxW) depth to 3x480x640 uint8 (triplicated)."""
    if depth_hwc.ndim == 3 and depth_hwc.shape[-1] == 1:
        depth_hwc = depth_hwc[..., 0]
    if depth_hwc.shape[:2] != (480, 640):
        depth_hwc = cv2.resize(depth_hwc, (640, 480), interpolation=cv2.INTER_NEAREST)
    depth_clipped = np.clip(depth_hwc, 0.0, max_m)
    depth_u8 = (depth_clipped / max_m * 255.0).astype(np.uint8)
    return np.stack([depth_u8, depth_u8, depth_u8], axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--repo_id", type=str, default="a2_pickplace_annotated_ee")
    parser.add_argument("--task", type=str, default="place the can in the tray")
    parser.add_argument("--lerobot_home", type=str, default=None)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    if args.lerobot_home:
        os.environ["HF_LEROBOT_HOME"] = args.lerobot_home

    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

    src = Path(args.input_file)
    if not src.exists():
        raise FileNotFoundError(src)

    # Peek first episode to learn dimensions and detect cameras.
    with h5py.File(src, "r") as f:
        if "data" in f:
            ep_grp = f["data"]
        elif "failed" in f:
            ep_grp = f["failed"]
        else:
            raise ValueError(f"Cannot find 'data' or 'failed' group in {src}")

        first_key = sorted(ep_grp.keys(), key=lambda k: int(k.split("_")[-1]))[0]
        first_ep = ep_grp[first_key]
        obs = first_ep["obs"]

        state_dim = int(obs["robot_joint_pos"].shape[1]) + 14
        action_dim = int(first_ep["actions"].shape[1])

        # Check for cameras.
        has_head_rgb = "head_camera_rgb" in obs
        has_chest_l = "chest_left_camera_rgb" in obs
        has_chest_r = "chest_right_camera_rgb" in obs
        has_depth = "head_camera_depth" in obs

        has_cameras = has_head_rgb and has_chest_l and has_chest_r

        # Head camera resolution (for resize scaling info).
        head_h, head_w = obs["head_camera_rgb"].shape[1:3] if has_head_rgb else (480, 640)
        depth_h, depth_w = obs["head_camera_depth"].shape[1:3] if has_depth else (480, 640)

        print(f"[convert-annotated] state_dim={state_dim}, action_dim={action_dim}")
        print(f"[convert-annotated] head_cam={head_h}x{head_w}, depth={depth_h}x{depth_w}")
        print(f"[convert-annotated] cameras: rgb={'Y' if has_cameras else 'N'}, depth={'Y' if has_depth else 'N'}")

        demo_keys = sorted(ep_grp.keys(), key=lambda k: int(k.split("_")[-1]))
        print(f"[convert-annotated] {len(demo_keys)} episodes")

    if (HF_LEROBOT_HOME / args.repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / args.repo_id)

    state_names = [f"joint_{i}" for i in range(state_dim - 14)] + [
        "left_eef_pos_x", "left_eef_pos_y", "left_eef_pos_z",
        "left_eef_quat_x", "left_eef_quat_y", "left_eef_quat_z", "left_eef_quat_w",
        "right_eef_pos_x", "right_eef_pos_y", "right_eef_pos_z",
        "right_eef_quat_x", "right_eef_quat_y", "right_eef_quat_z", "right_eef_quat_w",
    ]

    features = {
        "observation.state": {
            "dtype": "float32", "shape": (state_dim,), "names": [state_names],
        },
        "action": {
            "dtype": "float32", "shape": (action_dim,), "names": [[f"a{i}" for i in range(action_dim)]],
        },
    }

    if has_cameras:
        for cam_key, cam_name in [("head_camera_rgb", "cam_high"),
                                    ("chest_left_camera_rgb", "cam_chest_left"),
                                    ("chest_right_camera_rgb", "cam_chest_right")]:
            features[f"observation.images.{cam_name}"] = {
                "dtype": "image", "shape": (3, 480, 640),
                "names": ["channels", "height", "width"],
            }
    else:
        print("[convert-annotated] WARNING: No cameras found — state-only dataset")

    if has_depth:
        features["observation.images.cam_high_depth"] = {
            "dtype": "image", "shape": (3, 480, 640),
            "names": ["channels", "height", "width"],
        }

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id, fps=args.fps, robot_type="a2_humanoid",
        features=features, use_videos=False,
        image_writer_processes=10, image_writer_threads=5,
    )

    with h5py.File(src, "r") as f:
        if "data" in f:
            ep_grp = f["data"]
        elif "failed" in f:
            ep_grp = f["failed"]
        else:
            raise ValueError(f"No 'data' or 'failed' group in {src}")

        for k in tqdm.tqdm(demo_keys):
            ep = ep_grp[k]
            obs = ep["obs"]

            # Build 67-dim state: joints + left EE + right EE.
            joint_pos = obs["robot_joint_pos"][:]
            left_eef_pos = obs["left_eef_pos"][:]
            left_eef_quat = obs["left_eef_quat"][:]
            right_eef_pos = obs["right_eef_pos"][:]
            right_eef_quat = obs["right_eef_quat"][:]
            state = np.concatenate([
                joint_pos,
                left_eef_pos, left_eef_quat,
                right_eef_pos, right_eef_quat,
            ], axis=-1).astype(np.float32)

            action = ep["actions"][:].astype(np.float32)
            T = state.shape[0]
            assert action.shape[0] == T, f"length mismatch in {k}: state={T} action={action.shape[0]}"

            for i in range(T):
                frame = {
                    "observation.state": torch.from_numpy(state[i]),
                    "action": torch.from_numpy(action[i]),
                }
                if has_cameras:
                    frame["observation.images.cam_high"] = resize_chw_480x640(
                        obs["head_camera_rgb"][i])
                    frame["observation.images.cam_chest_left"] = resize_chw_480x640(
                        obs["chest_left_camera_rgb"][i])
                    frame["observation.images.cam_chest_right"] = resize_chw_480x640(
                        obs["chest_right_camera_rgb"][i])
                if has_depth:
                    frame["observation.images.cam_high_depth"] = depth_to_chw_480x640(
                        obs["head_camera_depth"][i])
                dataset.add_frame(frame, task=args.task)
            dataset.save_episode()

    print(f"[convert-annotated] done -> {HF_LEROBOT_HOME / args.repo_id}")


if __name__ == "__main__":
    main()
