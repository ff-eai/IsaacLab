"""Convert Isaac Lab A2 HDF5 → LeRobot v2 dataset with EE pose in state and
both EE and joint commands in action.

State (67-d):
  obs/robot_joint_pos      [53] qpos in env joint order
  obs/left_eef_pos         [3]  world-frame xyz
  obs/left_eef_quat        [4]  wxyz
  obs/right_eef_pos        [3]
  obs/right_eef_quat       [4]

Action (53-d):
  data/actions[0:14]       [14] EE wrist target pose (3 pos + 4 quat × 2 hands)
  data/processed_actions[0:15] [15] post-IK joint targets (14 arm + 1 waist)
  data/actions[14:38]      [24] hand joint position targets (mimic-expanded)

Cameras unchanged: head + chest_left + chest_right RGB at 480x640, plus
optional head depth replicated to 3 channels.

Usage::

    python convert_isaac_a2_to_lerobot_ee.py \
        --input_file /home/wagner/2T/wagner/dataset/issac_placn/a2_pickplace_v2_generated.hdf5 \
        --joint_names_json /workspace/2T/a2_joint_names.json \
        --repo_id a2_pickplace_v2_ee
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
    if rgb_hwc.shape[:2] != (480, 640):
        rgb_hwc = cv2.resize(rgb_hwc, (640, 480), interpolation=cv2.INTER_AREA)
    return np.transpose(rgb_hwc, (2, 0, 1))


def depth_to_chw_480x640(depth: np.ndarray, max_m: float = 4.0) -> np.ndarray:
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.shape[:2] != (480, 640):
        depth = cv2.resize(depth, (640, 480), interpolation=cv2.INTER_NEAREST)
    depth = np.clip(depth, 0.0, max_m)
    depth_u8 = (depth / max_m * 255.0).astype(np.uint8)
    return np.stack([depth_u8, depth_u8, depth_u8], axis=0)


EE_COMPONENT_NAMES = [
    "left_eef_pos_x", "left_eef_pos_y", "left_eef_pos_z",
    "left_eef_quat_w", "left_eef_quat_x", "left_eef_quat_y", "left_eef_quat_z",
    "right_eef_pos_x", "right_eef_pos_y", "right_eef_pos_z",
    "right_eef_quat_w", "right_eef_quat_x", "right_eef_quat_y", "right_eef_quat_z",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--joint_names_json", type=str, required=True)
    parser.add_argument("--repo_id", type=str, default="a2_pickplace_v2_ee")
    parser.add_argument("--task", type=str, default="place the can in the tray")
    parser.add_argument("--lerobot_home", type=str, default=None)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    if args.lerobot_home:
        os.environ["HF_LEROBOT_HOME"] = args.lerobot_home

    try:
        from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
    except ModuleNotFoundError:
        from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

    with open(args.joint_names_json) as f:
        joint_names = json.load(f)["joint_names"]
    qpos_dim = len(joint_names)
    state_dim = qpos_dim + len(EE_COMPONENT_NAMES)
    print(f"[convert-ee] qpos_dim={qpos_dim}, state_dim={state_dim}")

    state_names = list(joint_names) + EE_COMPONENT_NAMES

    src = Path(args.input_file)
    if not src.exists():
        raise FileNotFoundError(src)

    with h5py.File(src, "r") as f:
        first_key = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))[0]
        raw_action_dim = int(f[f"data/{first_key}/actions"].shape[1])           # 38
        proc_action_dim = int(f[f"data/{first_key}/processed_actions"].shape[1]) # 39
        print(f"[convert-ee] raw_action_dim={raw_action_dim} proc_action_dim={proc_action_dim}")
    # Compose 53-d action: 14 EE pose + 15 IK joint targets + 24 hand qpos.
    action_dim = 14 + 15 + 24
    assert action_dim == 53

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
            "names": [[
                # 14 EE wrist pose
                "L_eef_tgt_pos_x", "L_eef_tgt_pos_y", "L_eef_tgt_pos_z",
                "L_eef_tgt_quat_w", "L_eef_tgt_quat_x", "L_eef_tgt_quat_y", "L_eef_tgt_quat_z",
                "R_eef_tgt_pos_x", "R_eef_tgt_pos_y", "R_eef_tgt_pos_z",
                "R_eef_tgt_quat_w", "R_eef_tgt_quat_x", "R_eef_tgt_quat_y", "R_eef_tgt_quat_z",
                # 15 IK joint targets
                "joint_tgt_arm_l_1", "joint_tgt_arm_l_2", "joint_tgt_arm_l_3",
                "joint_tgt_arm_l_4", "joint_tgt_arm_l_5", "joint_tgt_arm_l_6", "joint_tgt_arm_l_7",
                "joint_tgt_arm_r_1", "joint_tgt_arm_r_2", "joint_tgt_arm_r_3",
                "joint_tgt_arm_r_4", "joint_tgt_arm_r_5", "joint_tgt_arm_r_6", "joint_tgt_arm_r_7",
                "joint_tgt_waist",
                # 24 hand qpos targets
                *[f"hand_tgt_{i}" for i in range(24)],
            ]],
        },
    }
    for cam in ("cam_high", "cam_chest_left", "cam_chest_right"):
        features[f"observation.images.{cam}"] = {
            "dtype": "image", "shape": (3, 480, 640),
            "names": ["channels", "height", "width"],
        }
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
        demo_keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))
        print(f"[convert-ee] {len(demo_keys)} episodes -> {HF_LEROBOT_HOME / args.repo_id}")
        for k in tqdm.tqdm(demo_keys):
            d = f[f"data/{k}"]

            qpos = d["obs/robot_joint_pos"][:].astype(np.float32)
            l_pos = d["obs/left_eef_pos"][:].astype(np.float32)
            l_quat = d["obs/left_eef_quat"][:].astype(np.float32)
            r_pos = d["obs/right_eef_pos"][:].astype(np.float32)
            r_quat = d["obs/right_eef_quat"][:].astype(np.float32)
            state = np.concatenate([qpos, l_pos, l_quat, r_pos, r_quat], axis=1)
            raw_actions = d["actions"][:].astype(np.float32)              # (T, 38) = 14 EE + 24 hand
            proc_actions = d["processed_actions"][:].astype(np.float32)   # (T, 39) = 15 IK + 24 hand
            action = np.concatenate([
                raw_actions[:, :14],     # 14 EE wrist pose
                proc_actions[:, :15],    # 15 post-IK joint targets (14 arm + 1 waist)
                raw_actions[:, 14:38],   # 24 hand qpos targets
            ], axis=1)

            head_rgb = d["obs/head_camera_rgb"][:]
            chest_l = d["obs/chest_left_camera_rgb"][:]
            chest_r = d["obs/chest_right_camera_rgb"][:]
            head_depth = d["obs/head_camera_depth"][:] if "obs/head_camera_depth" in d else None

            T = state.shape[0]
            assert action.shape[0] == T
            assert state.shape[1] == state_dim, f"state {state.shape[1]} != {state_dim}"
            assert action.shape[1] == action_dim

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

    print("[convert-ee] done.")


if __name__ == "__main__":
    main()
