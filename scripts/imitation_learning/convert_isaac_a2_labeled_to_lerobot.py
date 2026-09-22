"""Convert labeled A2 failure HDF5 (capture_failure_dataset output) → LeRobot v2.

Reads episodes from the ``failed/`` group, optionally filters by failure labels,
samples up to N episodes per category, and exports:
  - 41-DOF action; observation.state is 41-DOF joints + 8 label floats
  - Label columns (indices 41-48): failure_type, hand_motion_result,
    failure_step, root_cause_step, recoverable, hand_motion_state,
    timestep_state, is_near_failure (see label_maps.json)

Requires 3 RGB cameras in each episode (head + chest_left + chest_right).
Use inject_cameras_labeled_failures.py if the source HDF5 was recorded without cameras.

Usage:
  python convert_isaac_a2_labeled_to_lerobot.py \\
      --input_file /path/to/a2_pickplace_failed_balanced_labeled.hdf5 \\
      --joint_names_json /path/to/a2_joint_names.json \\
      --repo_id a2_pickplace_failed_labeled_300 \\
      --max_per_filter 100
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path

import h5py
import numpy as np
import torch
import tqdm

from convert_isaac_a2_to_lerobot import (
    HAND_GROUPS,
    DIRECT_MAP,
    MOTOR_NAMES,
    build_projection,
    project_joints,
    resize_chw_480x640,
    resize_depth_chw_480x640,
)

FAILURE_TYPE_MAP = {
    "placed_outside_tray": 0,
    "unknown": 1,
}

HAND_MOTION_STATE_MAP = {
    "normal": 0,
    "approaching_open": 1,
    "closing": 2,
    "grasped": 3,
    "knocked": 4,
}

TIMESTEP_STATE_MAP = {
    "normal": 0,
    "degraded": 1,
    "unsafe": 2,
    "irreversible_failure": 3,
    "recovery_attempt": 4,
}

LABEL_FEATURE_NAMES = [
    "label.failure_type",
    "label.hand_motion_result",
    "label.failure_step",
    "label.root_cause_step",
    "label.recoverable",
    "label.hand_motion_state",
    "label.timestep_state",
    "label.is_near_failure",
]
NUM_LABEL_FEATURES = len(LABEL_FEATURE_NAMES)
STATE_DIM = 41

HAND_MOTION_RESULT_MAP = {
    "no_hand_engagement": 0,
    "closed_without_lift": 1,
    "grasp_without_closing": 2,
    "knocked_can_without_grasp": 3,
    "valid_grasp_path": 4,
}


def _decode_attr(val) -> str:
    if isinstance(val, bytes):
        return val.decode("utf-8")
    return str(val)


def _load_bool_series(ep, key: str, length: int) -> np.ndarray:
    if key not in ep:
        return np.zeros(length, dtype=np.int8)
    data = np.asarray(ep[key][:]).reshape(-1)
    n = min(length, len(data))
    out = np.zeros(length, dtype=np.int8)
    out[:n] = data[:n].astype(np.int8)
    return out


def _load_int8_label_series(ep, key: str, length: int, mapping: dict[str, int]) -> np.ndarray:
    if key not in ep:
        return np.zeros(length, dtype=np.int8)
    raw = ep[key][:]
    if raw.dtype.kind in ("S", "U", "O"):
        labels = [_decode_attr(x).strip() for x in raw.reshape(-1)]
        return np.array([mapping.get(lb, 0) for lb in labels[:length]], dtype=np.int8)
    data = np.asarray(raw).reshape(-1)
    n = min(length, len(data))
    out = np.zeros(length, dtype=np.int8)
    out[:n] = data[:n].astype(np.int8)
    return out


def collect_episode_keys(
    h5f: h5py.File,
    *,
    failure_type: str | None = None,
    hand_motion_result: str | None = None,
) -> list[str]:
    keys = []
    for key in h5f["failed"].keys():
        ep = h5f[f"failed/{key}"]
        ft = _decode_attr(ep.attrs.get("failure_type", ""))
        hm = _decode_attr(ep.attrs.get("hand_motion_result", ""))
        if failure_type is not None and ft != failure_type:
            continue
        if hand_motion_result is not None and hm != hand_motion_result:
            continue
        keys.append(key)
    return sorted(keys, key=lambda k: int(k.split("_")[-1]))


def sample_disjoint_episodes(
    h5f: h5py.File,
    specs: list[tuple[str, str | None, str | None]],
    max_per: int,
    seed: int,
) -> list[tuple[str, str, str]]:
    """Return list of (episode_key, failure_type, hand_motion_result) with disjoint keys when possible."""
    rng = random.Random(seed)
    used: set[str] = set()
    selected: list[tuple[str, str, str]] = []

    for label_name, ft_filter, hm_filter in specs:
        pool = collect_episode_keys(h5f, failure_type=ft_filter, hand_motion_result=hm_filter)
        pool = [k for k in pool if k not in used]
        rng.shuffle(pool)
        take = pool[:max_per]
        for key in take:
            ep = h5f[f"failed/{key}"]
            used.add(key)
            selected.append(
                (
                    key,
                    _decode_attr(ep.attrs.get("failure_type", "")),
                    _decode_attr(ep.attrs.get("hand_motion_result", "")),
                )
            )
        print(f"[convert-labeled] {label_name}: selected {len(take)}/{max_per} (pool={len(pool)})")
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--joint_names_json", type=str, required=True)
    parser.add_argument("--repo_id", type=str, default="a2_pickplace_failed_labeled")
    parser.add_argument("--task", type=str, default="place the can in the tray")
    parser.add_argument("--lerobot_home", type=str, default=None)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max_per_filter", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--export_placed_outside_tray",
        action="store_true",
        default=True,
        help="Include 100 episodes with failure_type=placed_outside_tray",
    )
    parser.add_argument(
        "--export_no_hand_engagement",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--export_closed_without_lift",
        action="store_true",
        default=True,
    )
    args = parser.parse_args()

    if args.lerobot_home:
        os.environ["HF_LEROBOT_HOME"] = args.lerobot_home

    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

    with open(args.joint_names_json) as f:
        joint_names = json.load(f)["joint_names"]
    direct_pairs, avg_groups = build_projection(joint_names)

    src = Path(args.input_file)
    if not src.exists():
        raise FileNotFoundError(src)

    specs: list[tuple[str, str | None, str | None]] = []
    if args.export_closed_without_lift:
        specs.append(("closed_without_lift", None, "closed_without_lift"))
    if args.export_no_hand_engagement:
        specs.append(("no_hand_engagement", None, "no_hand_engagement"))
    if args.export_placed_outside_tray:
        specs.append(("placed_outside_tray", "placed_outside_tray", None))

    with h5py.File(src, "r") as h5f:
        episodes = sample_disjoint_episodes(h5f, specs, args.max_per_filter, args.seed)

    if not episodes:
        raise RuntimeError("No episodes selected for export")

    out_dir = HF_LEROBOT_HOME / args.repo_id
    if out_dir.exists():
        shutil.rmtree(out_dir)

    state_names = list(MOTOR_NAMES) + LABEL_FEATURE_NAMES
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (STATE_DIM + NUM_LABEL_FEATURES,),
            "names": [state_names],
        },
        "action": {"dtype": "float32", "shape": (STATE_DIM,), "names": [MOTOR_NAMES]},
    }

    required_cams = ("head_camera_rgb", "chest_left_camera_rgb", "chest_right_camera_rgb")
    with h5py.File(src, "r") as h5f:
        first = h5f[f"failed/{episodes[0][0]}"]
        obs_keys = list(first["obs"].keys())
        missing = [k for k in required_cams if k not in obs_keys]
        if missing:
            raise RuntimeError(
                f"Input HDF5 missing camera obs {missing}. "
                "Run inject_cameras_labeled_failures.py first."
            )
        has_depth = "head_camera_depth" in obs_keys

    for cam in ("cam_high", "cam_chest_left", "cam_chest_right"):
        features[f"observation.images.{cam}"] = {
            "dtype": "image",
            "shape": (3, 480, 640),
            "names": ["channels", "height", "width"],
        }
    if has_depth:
        features["observation.depth.cam_high"] = {
            "dtype": "float32",
            "shape": (1, 480, 640),
            "names": ["channel", "height", "width"],
        }

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        robot_type="a2_humanoid",
        features=features,
        use_videos=False,
        image_writer_processes=4,
        image_writer_threads=2,
    )

    label_meta = {
        "failure_type_map": FAILURE_TYPE_MAP,
        "hand_motion_result_map": HAND_MOTION_RESULT_MAP,
        "hand_motion_state_map": HAND_MOTION_STATE_MAP,
        "timestep_state_map": TIMESTEP_STATE_MAP,
    }
    meta_path = out_dir / "label_maps.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(label_meta, f, indent=2)

    with h5py.File(src, "r") as h5f:
        print(f"[convert-labeled] Exporting {len(episodes)} episodes -> {out_dir}")
        for key, ft, hm in tqdm.tqdm(episodes):
            ep = h5f[f"failed/{key}"]
            joint_pos = ep["obs/robot_joint_pos"][:]
            if joint_pos.shape[1] == 41:
                state = joint_pos.astype(np.float32)
            else:
                state = project_joints(joint_pos, direct_pairs, avg_groups)
            action = np.empty_like(state)
            action[:-1] = state[1:]
            action[-1] = state[-1]
            T = state.shape[0]

            ft_id = FAILURE_TYPE_MAP.get(ft, FAILURE_TYPE_MAP["unknown"])
            hm_id = HAND_MOTION_RESULT_MAP.get(hm, 0)
            failure_step = int(ep.attrs.get("failure_step", -1))
            root_cause_step = int(ep.attrs.get("root_cause_step", -1))
            recoverable = int(ep.attrs.get("recoverable", 0))

            hms = _load_int8_label_series(ep, "hand_motion_state", T, HAND_MOTION_STATE_MAP)
            tss = _load_int8_label_series(ep, "timestep_state", T, TIMESTEP_STATE_MAP)
            near = _load_bool_series(ep, "is_near_failure", T)

            head_rgb = ep["obs/head_camera_rgb"][:]
            chest_l = ep["obs/chest_left_camera_rgb"][:]
            chest_r = ep["obs/chest_right_camera_rgb"][:]
            head_depth = ep["obs/head_camera_depth"][:] if has_depth else None

            for i in range(T):
                label_vec = np.array(
                    [
                        ft_id,
                        hm_id,
                        failure_step,
                        root_cause_step,
                        recoverable,
                        hms[i],
                        tss[i],
                        near[i],
                    ],
                    dtype=np.float32,
                )
                obs_state = np.concatenate([state[i], label_vec])
                frame = {
                    "observation.state": torch.from_numpy(obs_state),
                    "action": torch.from_numpy(action[i]),
                    "task": args.task,
                }
                frame["observation.images.cam_high"] = resize_chw_480x640(head_rgb[i])
                frame["observation.images.cam_chest_left"] = resize_chw_480x640(chest_l[i])
                frame["observation.images.cam_chest_right"] = resize_chw_480x640(chest_r[i])
                if has_depth:
                    frame["observation.depth.cam_high"] = resize_depth_chw_480x640(head_depth[i])
                dataset.add_frame(frame)
            dataset.save_episode()

    print(f"[convert-labeled] Done. {len(episodes)} episodes at {out_dir}")
    print(f"[convert-labeled] Label maps: {meta_path}")


if __name__ == "__main__":
    main()
