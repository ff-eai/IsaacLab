"""Convert Isaac Mimic annotated HDF5 -> LeRobot v2 with pi0.7-style conditioning.

Adds per-frame multimodal prompts inspired by the pi0.7 blog:
  - task_instruction (episode-level language)
  - subtask_instruction (per-frame, from mimic subtask boundaries)
  - observation.images.subgoal (visual subgoal at current subtask end)
  - prompt.metadata (quality, speed episode scalars)
  - prompt.control_modality (eef vs joint)

Subtask boundaries are inferred from ``obs/datagen_info/subtask_term_signals``
(e.g. ``idle_right``) when present; otherwise from configured instruction text
with a single subtask spanning the episode.

Run:
  python convert_annotated_to_pi07_lerobot.py \\
      --input_file /path/to/a2_pickplace_v2_annotated.hdf5 \\
      --repo_id a2_pickplace_pi07 \\
      --lerobot_home /path/to/lerobot_cache
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import tqdm

# Default language prompts for A2 place-can-into-tray (matches pickplace_a2_mimic_env_cfg).
DEFAULT_TASK_INSTRUCTION = "place the can in the tray"
DEFAULT_SUBTASKS = (
    {"instruction": "grasp the can", "term_signal": "idle_right"},
    {"instruction": "place the can in the tray", "term_signal": None},
)


def resize_chw_480x640(rgb_hwc: np.ndarray) -> np.ndarray:
    if rgb_hwc.shape[:2] != (480, 640):
        rgb_hwc = cv2.resize(rgb_hwc, (640, 480), interpolation=cv2.INTER_AREA)
    return np.transpose(rgb_hwc, (2, 0, 1))


def depth_to_chw_480x640(depth_hwc: np.ndarray, max_m: float = 4.0) -> np.ndarray:
    if depth_hwc.ndim == 3 and depth_hwc.shape[-1] == 1:
        depth_hwc = depth_hwc[..., 0]
    if depth_hwc.shape[:2] != (480, 640):
        depth_hwc = cv2.resize(depth_hwc, (640, 480), interpolation=cv2.INTER_NEAREST)
    depth_clipped = np.clip(depth_hwc, 0.0, max_m)
    depth_u8 = (depth_clipped / max_m * 255.0).astype(np.uint8)
    return np.stack([depth_u8, depth_u8, depth_u8], axis=0)


@dataclass
class SubtaskSpan:
    instruction: str
    start: int
    end: int  # exclusive
    subgoal_index: int


def _first_rising_edge(signal: np.ndarray) -> int | None:
    if signal.size == 0:
        return None
    diff = np.diff(signal.astype(np.int8))
    rising = np.where(diff > 0)[0]
    if rising.size == 0:
        return None
    return int(rising[0] + 1)


def build_subtask_spans(
    length: int,
    subtask_defs: tuple[dict, ...],
    term_signals: dict[str, np.ndarray] | None,
) -> list[SubtaskSpan]:
    """Split an episode into subtask spans using mimic termination signals."""
    boundaries = [0]
    for spec in subtask_defs[:-1]:
        term = spec.get("term_signal")
        if term is None or term_signals is None or term not in term_signals:
            boundaries.append(length)
            break
        edge = _first_rising_edge(term_signals[term][:length])
        boundaries.append(length if edge is None else edge)
    boundaries.append(length)

    # Ensure monotonic, non-empty spans.
    cleaned = [0]
    for b in boundaries[1:]:
        b = max(cleaned[-1], min(int(b), length))
        if b > cleaned[-1]:
            cleaned.append(b)
    if cleaned[-1] != length:
        cleaned.append(length)

    spans: list[SubtaskSpan] = []
    for i, spec in enumerate(subtask_defs):
        if i >= len(cleaned) - 1:
            break
        start, end = cleaned[i], cleaned[i + 1]
        if start >= end:
            continue
        spans.append(
            SubtaskSpan(
                instruction=spec["instruction"],
                start=start,
                end=end,
                subgoal_index=min(end - 1, length - 1),
            )
        )
    if not spans:
        spans.append(
            SubtaskSpan(
                instruction=subtask_defs[-1]["instruction"],
                start=0,
                end=length,
                subgoal_index=max(0, length - 1),
            )
        )
    return spans


def span_for_frame(spans: list[SubtaskSpan], frame_idx: int) -> SubtaskSpan:
    for span in spans:
        if span.start <= frame_idx < span.end:
            return span
    return spans[-1]


def episode_quality(ep) -> float:
    if "success" in ep.attrs:
        return 1.0 if bool(ep.attrs["success"]) else 0.35
    if "failure_reasons" in ep.attrs:
        try:
            reasons = json.loads(ep.attrs["failure_reasons"])
            if "quality_score" in reasons:
                return float(reasons["quality_score"])
        except (json.JSONDecodeError, TypeError):
            pass
    return 0.5


def episode_speed(length: int, median_length: float, fps: int) -> float:
    """Normalized speed in [0, 1]: faster episodes (fewer steps) score higher."""
    if median_length <= 0:
        return 0.5
    ratio = median_length / max(float(length), 1.0)
    return float(np.clip(ratio, 0.2, 1.0))


def load_subtask_defs(path: Path | None) -> tuple[dict, ...]:
    if path is None:
        return DEFAULT_SUBTASKS
    data = json.loads(path.read_text())
    return tuple(data["subtasks"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--repo_id", type=str, default="a2_pickplace_pi07")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK_INSTRUCTION)
    parser.add_argument("--lerobot_home", type=str, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--control_modality",
        type=str,
        default="eef",
        choices=["eef", "joint"],
        help="Control modality label stored per frame (pi0.7 conditioning).",
    )
    parser.add_argument(
        "--subtask_config",
        type=str,
        default=None,
        help="Optional JSON with {\"subtasks\": [{\"instruction\": ..., \"term_signal\": ...}, ...]}",
    )
    parser.add_argument(
        "--group",
        type=str,
        default="data",
        choices=["data", "failed", "auto"],
        help="HDF5 group to convert. 'auto' merges data then failed.",
    )
    args = parser.parse_args()

    if args.lerobot_home:
        os.environ["HF_LEROBOT_HOME"] = args.lerobot_home

    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

    src = Path(args.input_file)
    if not src.exists():
        raise FileNotFoundError(src)

    subtask_defs = load_subtask_defs(Path(args.subtask_config) if args.subtask_config else None)

    with h5py.File(src, "r") as f:
        groups = []
        if args.group == "auto":
            if "data" in f:
                groups.append("data")
            if "failed" in f:
                groups.append("failed")
        else:
            groups.append(args.group)

        demo_keys: list[tuple[str, str]] = []
        lengths: list[int] = []
        for grp in groups:
            if grp not in f:
                continue
            for k in sorted(f[grp].keys(), key=lambda x: int(x.split("_")[-1])):
                demo_keys.append((grp, k))
                lengths.append(int(f[grp][k].attrs.get("num_samples", f[grp][k]["actions"].shape[0])))

        if not demo_keys:
            raise ValueError(f"No episodes found in {src}")

        first_ep = f[demo_keys[0][0]][demo_keys[0][1]]
        obs0 = first_ep["obs"]
        state_dim = int(obs0["robot_joint_pos"].shape[1]) + 14
        action_dim = int(first_ep["actions"].shape[1])
        has_cameras = all(k in obs0 for k in ("head_camera_rgb", "chest_left_camera_rgb", "chest_right_camera_rgb"))
        has_depth = "head_camera_depth" in obs0
        median_length = float(np.median(lengths))

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
        "prompt.task_instruction": {"dtype": "string", "shape": (1,), "names": None},
        "prompt.subtask_instruction": {"dtype": "string", "shape": (1,), "names": None},
        "prompt.metadata": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["quality", "speed"],
        },
        "prompt.control_modality": {"dtype": "string", "shape": (1,), "names": None},
    }

    if has_cameras:
        for cam_name in ("cam_high", "cam_chest_left", "cam_chest_right", "subgoal"):
            features[f"observation.images.{cam_name}"] = {
                "dtype": "image", "shape": (3, 480, 640),
                "names": ["channels", "height", "width"],
            }
    if has_depth:
        features["observation.images.cam_high_depth"] = {
            "dtype": "image", "shape": (3, 480, 640),
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
        for grp, k in tqdm.tqdm(demo_keys, desc="episodes"):
            ep = f[grp][k]
            obs = ep["obs"]
            T = int(ep.attrs.get("num_samples", ep["actions"].shape[0]))

            term_signals = None
            if "datagen_info" in obs and "subtask_term_signals" in obs["datagen_info"]:
                sig_grp = obs["datagen_info"]["subtask_term_signals"]
                term_signals = {name: sig_grp[name][:] for name in sig_grp.keys()}

            spans = build_subtask_spans(T, subtask_defs, term_signals)
            subgoal_cache = {
                span.subgoal_index: resize_chw_480x640(obs["head_camera_rgb"][span.subgoal_index])
                for span in spans
                if has_cameras
            }

            joint_pos = obs["robot_joint_pos"][:T]
            state = np.concatenate(
                [
                    joint_pos,
                    obs["left_eef_pos"][:T],
                    obs["left_eef_quat"][:T],
                    obs["right_eef_pos"][:T],
                    obs["right_eef_quat"][:T],
                ],
                axis=-1,
            ).astype(np.float32)
            action = ep["actions"][:T].astype(np.float32)

            quality = episode_quality(ep)
            speed = episode_speed(T, median_length, args.fps)

            for i in range(T):
                span = span_for_frame(spans, i)
                frame = {
                    "observation.state": torch.from_numpy(state[i]),
                    "action": torch.from_numpy(action[i]),
                    "prompt.task_instruction": args.task,
                    "prompt.subtask_instruction": span.instruction,
                    "prompt.metadata": np.array([quality, speed], dtype=np.float32),
                    "prompt.control_modality": args.control_modality,
                }
                if has_cameras:
                    frame["observation.images.cam_high"] = resize_chw_480x640(obs["head_camera_rgb"][i])
                    frame["observation.images.cam_chest_left"] = resize_chw_480x640(obs["chest_left_camera_rgb"][i])
                    frame["observation.images.cam_chest_right"] = resize_chw_480x640(obs["chest_right_camera_rgb"][i])
                    frame["observation.images.subgoal"] = subgoal_cache[span.subgoal_index]
                if has_depth:
                    frame["observation.images.cam_high_depth"] = depth_to_chw_480x640(obs["head_camera_depth"][i])
                frame["task"] = args.task
                dataset.add_frame(frame)
            dataset.save_episode()

    print(f"[convert-pi07] {len(demo_keys)} episodes -> {HF_LEROBOT_HOME / args.repo_id}")


if __name__ == "__main__":
    main()
