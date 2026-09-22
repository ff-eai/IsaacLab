"""Convert Isaac Lab Mimic-generated A2 HDF5 → LeRobot v2 dataset (RAW schema).

Same camera layout as `convert_isaac_a2_to_lerobot.py`, but the state and
action come straight from the recorded hdf5 instead of being projected to
the 41-DOF compact AgiBot motor space:

  - observation.state  := obs/robot_joint_pos  (53 floats — full A2 articulation)
  - action             := data/actions         (38 floats — Pink IK wrist
                          target poses 7+7 + 24 OmniHand/s6 joint targets)
  - observation.images.cam_high       := obs/head_camera_rgb       (resized to 480x640 CHW, uint8)
  - observation.images.cam_chest_left := obs/chest_left_camera_rgb (resized to 480x640 CHW, uint8)
  - observation.images.cam_chest_right:= obs/chest_right_camera_rgb (resized to 480x640 CHW, uint8)
  - observation.images.cam_high_depth := obs/head_camera_depth     (resized to 480x640 1-CHW, depth in metres normalised to uint8 by clipping to [0, 4]m and scaling × 64)

The `--joint_names_json` mapping is consumed only to validate the joint count
and to populate the `names` field of the 53-d state column for downstream tools.
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
    """Resize a HxWxC uint8 RGB frame to 480x640, then HWC→CHW."""
    if rgb_hwc.shape[:2] != (480, 640):
        rgb_hwc = cv2.resize(rgb_hwc, (640, 480), interpolation=cv2.INTER_AREA)
    return np.transpose(rgb_hwc, (2, 0, 1))


def depth_to_chw_480x640(depth: np.ndarray, max_m: float = 4.0) -> np.ndarray:
    """Convert a HxWx1 (or HxW) depth-in-metres frame → 3xHxW uint8 image
    (depth replicated across all 3 channels — single-channel images don't
    survive lerobot's PNG writer cleanly, so we ride the RGB path).
    Clipped to [0, max_m] m and scaled to [0, 255]; downstream training can
    rescale to metres by `depth_m = (px / 255) * max_m` (using any channel)."""
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.shape[:2] != (480, 640):
        depth = cv2.resize(depth, (640, 480), interpolation=cv2.INTER_NEAREST)
    depth = np.clip(depth, 0.0, max_m)
    depth_u8 = (depth / max_m * 255.0).astype(np.uint8)
    return np.stack([depth_u8, depth_u8, depth_u8], axis=0)  # (3, 480, 640)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--joint_names_json", type=str, required=True)
    parser.add_argument("--repo_id", type=str, default="a2_pickplace_isaac_raw")
    parser.add_argument("--task", type=str, default="place the can in the tray")
    parser.add_argument(
        "--lerobot_home",
        type=str,
        default=None,
        help="Override HF_LEROBOT_HOME (defaults to lerobot's own default).",
    )
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    if args.lerobot_home:
        os.environ["HF_LEROBOT_HOME"] = args.lerobot_home

    # imported AFTER env override so HF_LEROBOT_HOME is honored
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

    with open(args.joint_names_json) as f:
        joint_names = json.load(f)["joint_names"]
    state_dim = len(joint_names)
    print(f"[convert-raw] {state_dim} env joints loaded")

    src = Path(args.input_file)
    if not src.exists():
        raise FileNotFoundError(src)

    # Peek the first demo to learn the action width.
    with h5py.File(src, "r") as f:
        first_key = sorted(
            f["data"].keys(), key=lambda k: int(k.split("_")[-1])
        )[0]
        action_dim = int(f[f"data/{first_key}/actions"].shape[1])
        print(f"[convert-raw] action dim from hdf5: {action_dim}")

    if (HF_LEROBOT_HOME / args.repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / args.repo_id)
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": [joint_names],
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": [[f"a{i}" for i in range(action_dim)]],
        },
    }
    for cam in ("cam_high", "cam_chest_left", "cam_chest_right"):
        features[f"observation.images.{cam}"] = {
            "dtype": "image",
            "shape": (3, 480, 640),
            "names": ["channels", "height", "width"],
        }
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
        demo_keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))
        print(f"[convert-raw] {len(demo_keys)} episodes -> {HF_LEROBOT_HOME / args.repo_id}")
        for k in tqdm.tqdm(demo_keys):
            d = f[f"data/{k}"]

            state = d["obs/robot_joint_pos"][:].astype(np.float32)
            action = d["actions"][:].astype(np.float32)
            head_rgb = d["obs/head_camera_rgb"][:]
            chest_l = d["obs/chest_left_camera_rgb"][:]
            chest_r = d["obs/chest_right_camera_rgb"][:]
            head_depth = d["obs/head_camera_depth"][:] if "obs/head_camera_depth" in d else None

            T = state.shape[0]
            assert action.shape[0] == T, (
                f"length mismatch in {k}: state={T} action={action.shape[0]}"
            )
            assert state.shape[1] == state_dim, (
                f"state dim {state.shape[1]} != expected {state_dim} in {k}"
            )
            assert action.shape[1] == action_dim, (
                f"action dim {action.shape[1]} != expected {action_dim} in {k}"
            )

            for i in range(T):
                frame = {
                    "observation.state": torch.from_numpy(state[i]),
                    "action": torch.from_numpy(action[i]),
                    "observation.images.cam_high": resize_chw_480x640(head_rgb[i]),
                    "observation.images.cam_chest_left": resize_chw_480x640(chest_l[i]),
                    "observation.images.cam_chest_right": resize_chw_480x640(chest_r[i]),
                    "task": args.task,
                }
                if head_depth is not None:
                    frame["observation.images.cam_high_depth"] = depth_to_chw_480x640(head_depth[i])
                else:
                    frame["observation.images.cam_high_depth"] = np.zeros((3, 480, 640), dtype=np.uint8)
                dataset.add_frame(frame)
            dataset.save_episode()

    print("[convert-raw] done.")


if __name__ == "__main__":
    main()
