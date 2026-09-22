"""Convert Isaac Lab Mimic-generated A2 HDF5 → LeRobot v2 dataset.

Produces a 41-DOF compact joint state (folds 12 finger joints per hand into 6
by averaging same-finger joint groups) and three RGB camera streams matching
the layout used by `configs/vla/a2_sapien.yaml`:
  - observation.images.cam_high       (from obs/head_camera_rgb)
  - observation.images.cam_chest_left  (from obs/chest_left_camera_rgb)
  - observation.images.cam_chest_right (from obs/chest_right_camera_rgb)

Action[t] = state[t+1] in 41-DOF (next-step joint targets), matching the BC
convention used by RoboTwin's existing pi0 converter.

Run inside the lingbot1 docker container (or any env with lerobot+h5py+cv2).
Joint name → index mapping is read from a JSON written by
`dump_a2_joint_names.py` so we don't need isaacsim in this step.

Usage (inside lingbot1 container, /workspace = ~/code on host):
  python /workspace/ext/code/IsaacLab/scripts/imitation_learning/convert_isaac_a2_to_lerobot.py \\
      --input_file /workspace/ext/issac/a2_pickplace_generated.hdf5 \\
      --joint_names_json /workspace/ext/issac/a2_joint_names.json \\
      --repo_id a2_pickplace_isaac \\
      --task "place the can in the tray"
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


# 41-DOF target layout — same field order as
# RoboTwin/policy/pi0/examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py
MOTOR_NAMES = [
    # left hand (6) — 12 source finger joints folded into 6 by averaging
    "left_thumb_swing", "left_thumb_main", "left_index", "left_middle", "left_ring", "left_pinky",
    # right hand (6)
    "right_thumb_swing", "right_thumb_main", "right_index", "right_middle", "right_ring", "right_pinky",
    # left arm (7)
    "left_arm_joint1", "left_arm_joint2", "left_arm_joint3",
    "left_arm_joint4", "left_arm_joint5", "left_arm_joint6", "left_arm_joint7",
    # right arm (7)
    "right_arm_joint1", "right_arm_joint2", "right_arm_joint3",
    "right_arm_joint4", "right_arm_joint5", "right_arm_joint6", "right_arm_joint7",
    # waist (1)
    "waist_yaw",
    # left leg (6)
    "left_hip_roll", "left_hip_yaw", "left_hip_pitch",
    "left_tarsus", "left_toe_pitch", "left_toe_roll",
    # right leg (6)
    "right_hip_roll", "right_hip_yaw", "right_hip_pitch",
    "right_tarsus", "right_toe_pitch", "right_toe_roll",
    # head (2)
    "head_joint1", "head_joint2",
]
assert len(MOTOR_NAMES) == 41, len(MOTOR_NAMES)

# (output joint name) → (source joint name in the IsaacLab env)
DIRECT_MAP = {
    "left_arm_joint1":  "idx13_left_arm_joint1",
    "left_arm_joint2":  "idx14_left_arm_joint2",
    "left_arm_joint3":  "idx15_left_arm_joint3",
    "left_arm_joint4":  "idx16_left_arm_joint4",
    "left_arm_joint5":  "idx17_left_arm_joint5",
    "left_arm_joint6":  "idx18_left_arm_joint6",
    "left_arm_joint7":  "idx19_left_arm_joint7",
    "right_arm_joint1": "idx20_right_arm_joint1",
    "right_arm_joint2": "idx21_right_arm_joint2",
    "right_arm_joint3": "idx22_right_arm_joint3",
    "right_arm_joint4": "idx23_right_arm_joint4",
    "right_arm_joint5": "idx24_right_arm_joint5",
    "right_arm_joint6": "idx25_right_arm_joint6",
    "right_arm_joint7": "idx26_right_arm_joint7",
    "waist_yaw":        "waist_yaw_joint",
    "left_hip_roll":    "idx01_left_hip_roll",
    "left_hip_yaw":     "idx02_left_hip_yaw",
    "left_hip_pitch":   "idx03_left_hip_pitch",
    "left_tarsus":      "idx04_left_tarsus",
    "left_toe_pitch":   "idx05_left_toe_pitch",
    "left_toe_roll":    "idx06_left_toe_roll",
    "right_hip_roll":   "idx07_right_hip_roll",
    "right_hip_yaw":    "idx08_right_hip_yaw",
    "right_hip_pitch":  "idx09_right_hip_pitch",
    "right_tarsus":     "idx10_right_tarsus",
    "right_toe_pitch":  "idx11_right_toe_pitch",
    "right_toe_roll":   "idx12_right_toe_roll",
    "head_joint1":      "idx27_head_joint1",
    "head_joint2":      "idx28_head_joint2",
}

# Hand fold: 6 source finger joints per hand → 6 outputs. Since A2_nomimic.usd
# now locks *_2 / *_3 joints (PhysicsFixedJoint), each finger exposes only its
# *_1 actuator. The averaging is effectively a pass-through; kept as a 1-element
# group for layout parity with the older 12→6 capture.
HAND_GROUPS = {
    "left_thumb_swing": ["L_thumb_swing_joint"],
    "left_thumb_main":  ["L_thumb_1_joint"],
    "left_index":       ["L_index_1_joint"],
    "left_middle":      ["L_middle_1_joint"],
    "left_ring":        ["L_ring_1_joint"],
    "left_pinky":       ["L_pinky_1_joint"],
    "right_thumb_swing":["R_thumb_swing_joint"],
    "right_thumb_main": ["R_thumb_1_joint"],
    "right_index":      ["R_index_1_joint"],
    "right_middle":     ["R_middle_1_joint"],
    "right_ring":       ["R_ring_1_joint"],
    "right_pinky":      ["R_pinky_1_joint"],
}


def build_projection(joint_names: list[str]):
    name_to_idx = {n: i for i, n in enumerate(joint_names)}
    direct_pairs = []
    for out_name, src_name in DIRECT_MAP.items():
        if src_name not in name_to_idx:
            raise KeyError(f"joint '{src_name}' missing from env joint list")
        direct_pairs.append((MOTOR_NAMES.index(out_name), name_to_idx[src_name]))
    avg_groups = []
    for out_name, src_names in HAND_GROUPS.items():
        idxs = [name_to_idx[n] for n in src_names]
        avg_groups.append((MOTOR_NAMES.index(out_name), idxs))
    return np.asarray(direct_pairs, dtype=np.int64), avg_groups


def project_joints(joint_pos: np.ndarray, direct_pairs: np.ndarray, avg_groups) -> np.ndarray:
    T = joint_pos.shape[0]
    out = np.zeros((T, 41), dtype=np.float32)
    out[:, direct_pairs[:, 0]] = joint_pos[:, direct_pairs[:, 1]]
    for out_idx, src_idxs in avg_groups:
        out[:, out_idx] = joint_pos[:, src_idxs].mean(axis=1)
    return out


def resize_chw_480x640(rgb_hwc: np.ndarray) -> np.ndarray:
    if rgb_hwc.shape[:2] != (480, 640):
        rgb_hwc = cv2.resize(rgb_hwc, (640, 480), interpolation=cv2.INTER_AREA)
    return np.transpose(rgb_hwc, (2, 0, 1))


def resize_depth_chw_480x640(depth_hwc: np.ndarray) -> torch.Tensor:
    """depth_hwc: (H, W, 1) float32 → (1, 480, 640) float32 torch tensor."""
    if depth_hwc.ndim == 3:
        depth_hw = depth_hwc[..., 0]
    else:
        depth_hw = depth_hwc
    if depth_hw.shape[:2] != (480, 640):
        # INTER_NEAREST avoids smearing across depth discontinuities
        depth_hw = cv2.resize(depth_hw, (640, 480), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(depth_hw.astype(np.float32))[None, :, :]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--joint_names_json", type=str, required=True)
    parser.add_argument("--repo_id", type=str, default="a2_pickplace_isaac")
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
    print(f"[convert] {len(joint_names)} env joints loaded")
    direct_pairs, avg_groups = build_projection(joint_names)

    src = Path(args.input_file)
    if not src.exists():
        raise FileNotFoundError(src)

    if (HF_LEROBOT_HOME / args.repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / args.repo_id)
    features = {
        "observation.state": {"dtype": "float32", "shape": (41,), "names": [MOTOR_NAMES]},
        "action":            {"dtype": "float32", "shape": (41,), "names": [MOTOR_NAMES]},
    }
    for cam in ("cam_high", "cam_chest_left", "cam_chest_right"):
        features[f"observation.images.{cam}"] = {
            "dtype": "image",
            "shape": (3, 480, 640),
            "names": ["channels", "height", "width"],
        }
    # Head depth as float32 (1 channel). Stored as a regular tensor feature
    # since LeRobot's "image" dtype expects uint8 RGB. Omitted if not present.
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
        # Support both IsaacLab 'data' format and capture_failure_dataset.py 'failed' format
        if "data" in f:
            data_group = f["data"]
        elif "failed" in f:
            data_group = f["failed"]
        else:
            raise ValueError(f"Cannot find 'data' or 'failed' group in {src}")

        demo_keys = sorted(data_group.keys(), key=lambda k: int(k.split("_")[-1]))
        # Filter to episodes that have all required camera keys (for capture_failure_dataset.py format
        # where cameras were injected into a subset of episodes)
        required_cam_keys = {"head_camera_rgb", "chest_left_camera_rgb", "chest_right_camera_rgb"}
        available_keys = set(data_group[demo_keys[0]]["obs"].keys())
        if not required_cam_keys.issubset(available_keys):
            # Check which episodes have cameras
            cam_keys = [k for k in demo_keys if required_cam_keys.issubset(set(data_group[k]["obs"].keys()))]
            if cam_keys:
                demo_keys = cam_keys
                print(f"[convert] {len(demo_keys)} episodes with cameras found")
            else:
                raise ValueError("No episodes with all required cameras found")
        print(f"[convert] {len(demo_keys)} episodes -> {HF_LEROBOT_HOME / args.repo_id}")
        skipped = 0
        for k in tqdm.tqdm(demo_keys):
            d = data_group[k]
            obs = d["obs"]
            joint_pos = obs["robot_joint_pos"][:]
            has_depth = "head_camera_depth" in obs
            head_depth = obs["head_camera_depth"][:] if has_depth else None

            # If robot_joint_pos is already 41-DOF (capture_failure_dataset format),
            # use it directly. The old code expected 53-DOF and projected down.
            if joint_pos.shape[1] == 41:
                state = joint_pos
            else:
                state = project_joints(joint_pos, direct_pairs, avg_groups)
            head_rgb = obs["head_camera_rgb"][:]
            chest_l = obs["chest_left_camera_rgb"][:]
            chest_r = obs["chest_right_camera_rgb"][:]

            action = np.empty_like(state)
            action[:-1] = state[1:]
            action[-1] = state[-1]

            T = state.shape[0]
            for i in range(T):
                frame = {
                    "observation.state": torch.from_numpy(state[i]),
                    "action": torch.from_numpy(action[i]),
                    "observation.images.cam_high": resize_chw_480x640(head_rgb[i]),
                    "observation.images.cam_chest_left": resize_chw_480x640(chest_l[i]),
                    "observation.images.cam_chest_right": resize_chw_480x640(chest_r[i]),
                }
                if has_depth and head_depth is not None:
                    frame["observation.depth.cam_high"] = resize_depth_chw_480x640(head_depth[i])
                dataset.add_frame(frame, task=args.task)
            dataset.save_episode()

    print(f"[convert] done.")


if __name__ == "__main__":
    main()
