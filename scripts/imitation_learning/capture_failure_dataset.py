"""Analyze failed episodes from Isaac Lab Mimic generation and produce labeled failure datasets.

Reads `*_failed.hdf5` files (generated with generation_keep_failed=True and
FailureReasonRecorderCfg), computes per-episode and per-timestep failure labels,
and writes a structured labeled output for downstream failure analysis, reward
modeling, and recovery-policy training.

Usage:
  python capture_failure_dataset.py \\
      --input_file /path/to/a2_place_can_tray_D0_failed.hdf5 \\
      --output_file failure_labeled.hdf5 \\
      --env_name a2_pickplace

Runs inside any environment with h5py+tqcm (docker container or host).
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import tqdm as _tqdm

# ---------------------------------------------------------------------------
# Failure taxonomy: maps termination-term failure patterns to high-level types
# ---------------------------------------------------------------------------

FAILURE_TYPE_MAP = {
    frozenset(): "unknown",
    # Grasp-related
    frozenset(["hands_released"]): "grasp_held",
    frozenset(["lift_latched"]): "lift_failed",
    frozenset(["lift_latched", "hands_released"]): "grasp_lost",
    # Placement-related
    frozenset(["xy_in_tray"]): "missed_tray",
    frozenset(["on_surface_z"]): "placement_height_wrong",
    frozenset(["xy_in_tray", "on_surface_z"]): "missed_tray",
    frozenset(["hands_released", "lift_latched", "xy_in_tray"]): "placed_outside_tray",
    frozenset(["hands_released", "lift_latched", "on_surface_z"]): "object_dropped",
    frozenset(["hands_released", "lift_latched", "on_surface_z", "xy_in_tray"]): "placed_outside_tray",
    frozenset(["hands_released", "lift_latched", "xy_in_tray", "on_surface_z"]): "placed_outside_tray",
    # Hard failures from termination terms (inverted semantics: False = bad)
    frozenset(["object_dropping"]): "object_dropped_midway",
    frozenset(["time_out"]): "timeout",
    frozenset(["object_dropping", "time_out"]): "object_dropped_midway",
}

# Priority ordering for multi-failure episodes (most severe first)
SEVERITY_MAP = {
    "unknown": "high",
    "timeout": "low",
    "grasp_held": "medium",
    "grasp_lost": "high",
    "lift_failed": "medium",
    "object_dropped": "critical",
    "object_dropped_midway": "critical",
    "placed_outside_tray": "medium",
    "missed_tray": "medium",
    "placement_height_wrong": "medium",
    "placement_failure": "medium",
}

# Near-failure: termination term was "active" but didn't trigger the episode reset
# In Isaac Lab's termination manager, terms trigger on consecutive frames exceeding threshold.
# We treat a term as "near-failure" if it has ever been close to triggering.
# Since we only have the final snapshot, we use the failure_reasons as our signal:
# a term that is False in the final snapshot but whose absence doesn't guarantee
# it was always False → potential near-miss.

# ---------------------------------------------------------------------------
# Observation field helpers — what fields to extract from each episode group
# ---------------------------------------------------------------------------

# Fields to extract (flat list of leaf key names under each episode group)
REQUIRED_FIELDS = ["obs/joint_pos", "obs/head_camera_rgb", "obs/head_camera_depth"]
OPTIONAL_FIELDS = [
    "obs/chest_left_camera_rgb",
    "obs/chest_right_camera_rgb",
    "obs/left_wrist_camera_rgb",
    "obs/right_wrist_camera_rgb",
    "obs/target_eef_pose",
    "obs/object_pose",
    "obs/tray_pose",
    "actions",
    "states",
    "obs",
    "initial_state",
]

# Joint indices for the A2 (from convert_isaac_a2_to_lerobot.py MOTOR_NAMES)
A2_JOINT_NAMES = [
    # left hand (6)
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

# Gripper joint indices (thumb joints: left/right thumb_swing + thumb_main)
GRIPPER_INDICES = [0, 1, 12, 13]

HAND_MOTION_RESULTS = (
    "valid_grasp_path",
    "grasp_without_closing",
    "knocked_can_without_grasp",
    "closed_without_lift",
    "no_hand_engagement",
)

HAND_MOTION_STATE_LABELS = (
    "normal",
    "approaching_open",
    "closing",
    "grasped",
    "knocked",
)


# ---------------------------------------------------------------------------
# Trajectory-level analysis
# ---------------------------------------------------------------------------

def _infer_hands_near(
    episode_group: h5py.Group,
    grasp_threshold: float = 0.20,
) -> np.ndarray | None:
    """Hands near can: use min_hand_dist with grasp_threshold when available."""
    hand_dist = _get_field(episode_group, "labels/min_hand_dist")
    if hand_dist is not None:
        hand_dist = np.asarray(hand_dist).reshape(-1)
        return hand_dist < grasp_threshold

    near = _load_label_series(episode_group, "hands_near_can")
    if near is not None:
        return near
    return None


def _infer_gripper_closed(episode_group: h5py.Group) -> np.ndarray | None:
    """Gripper closed: prefer recorded label, else joint-pos proxy."""
    closed = _load_label_series(episode_group, "gripper_closed")
    if closed is not None:
        return closed

    joint_data = _get_field(episode_group, "obs/joint_pos")
    if joint_data is None or joint_data.ndim != 2 or joint_data.shape[1] <= max(GRIPPER_INDICES):
        return None
    gripper_pos = joint_data[:, GRIPPER_INDICES]
    return np.all(gripper_pos > 0.5, axis=1)


def _infer_can_knocked(
    episode_group: h5py.Group,
    hands_near: np.ndarray | None,
    lift: np.ndarray | None,
    gripper_closed: np.ndarray | None = None,
    spawn_ignore_steps: int = 5,
    knock_z_drop_m: float = 0.008,
    knock_xy_jerk_m: float = 0.008,
    knock_displacement_m: float = 0.003,
    knock_eef_speed_m: float = 0.003,
) -> np.ndarray | None:
    """Can knocked this step: prefer recorded label, else tuned displacement/eef proxy."""
    knocked = _load_label_series(episode_group, "can_knocked_now")
    if knocked is not None:
        return knocked

    if hands_near is None:
        return None

    obj_z = _get_field(episode_group, "labels/object_z")
    if obj_z is None:
        return None

    obj_z = np.asarray(obj_z).reshape(-1)
    n = min(len(hands_near), len(obj_z))
    if n <= 1:
        return np.zeros(n, dtype=bool)

    post = np.arange(n) > spawn_ignore_steps
    open_near = hands_near[:n]
    if gripper_closed is not None and len(gripper_closed) >= n:
        open_near = open_near & ~gripper_closed[:n]

    obj_pos = _get_field(episode_group, "obs/object_pos")
    if obj_pos is not None:
        obj_pos = np.asarray(obj_pos).reshape(n, -1)[:, :3]
        dpos = np.linalg.norm(np.diff(obj_pos, axis=0, prepend=obj_pos[:1]), axis=1)
        z_drop = (obj_pos[:, 2] - np.concatenate([[obj_pos[0, 2]], obj_pos[: n - 1, 2]])) < -knock_z_drop_m
        displaced = dpos > knock_displacement_m
    else:
        z_prev = np.concatenate([[obj_z[0]], obj_z[: n - 1]])
        z_drop = (obj_z[:n] - z_prev[:n]) < -knock_z_drop_m
        displaced = np.zeros(n, dtype=bool)

    xy_jerk = np.zeros(n, dtype=bool)
    xy_dist = _get_field(episode_group, "labels/xy_dist")
    if xy_dist is not None:
        xy_dist = np.asarray(xy_dist).reshape(-1)
        xy_prev = np.concatenate([[xy_dist[0]], xy_dist[: n - 1]])
        xy_jerk = np.abs(xy_dist[:n] - xy_prev[:n]) > knock_xy_jerk_m

    eef_fast = np.zeros(n, dtype=bool)
    eef_pos = _get_field(episode_group, "obs/right_eef_pos")
    if eef_pos is not None:
        eef_pos = np.asarray(eef_pos).reshape(n, -1)[:, :3]
        eef_speed = np.linalg.norm(np.diff(eef_pos, axis=0, prepend=eef_pos[:1]), axis=1)
        eef_fast = eef_speed > knock_eef_speed_m

    not_lifted = np.ones(n, dtype=bool)
    if lift is not None and len(lift) >= n:
        not_lifted = ~lift[:n]

    return post & open_near[:n] & not_lifted & (z_drop | xy_jerk | displaced | eef_fast)


def _infer_spawn_knock(
    episode_group: h5py.Group,
    spawn_ignore_steps: int = 5,
    knock_z_drop_m: float = 0.008,
) -> bool:
    """Early can settlement/drop before hand engages."""
    spawn = _load_label_series(episode_group, "spawn_can_knocked")
    if spawn is not None and bool(spawn.any()):
        return True

    obj_z = _get_field(episode_group, "labels/object_z")
    if obj_z is None:
        return False
    obj_z = np.asarray(obj_z).reshape(-1)
    early_end = min(len(obj_z), spawn_ignore_steps + 6)
    if early_end <= 1:
        return False
    dz = np.diff(obj_z[:early_end], prepend=obj_z[0])
    return bool((dz[1:early_end] < -knock_z_drop_m).any())


def _infer_pick_without_close(
    episode_group: h5py.Group,
    hands_near: np.ndarray | None,
    gripper_closed: np.ndarray | None,
    open_near_min_steps: int = 3,
) -> bool:
    """Pick/grasp attempt without closing the gripper."""
    recorded = _load_label_series(episode_group, "pick_without_close_now")
    if recorded is not None and bool(recorded.any()):
        return True

    if hands_near is None:
        return False
    n = len(hands_near)
    closed_near = np.zeros(n, dtype=bool)
    if gripper_closed is not None and len(gripper_closed) >= n:
        closed_near = hands_near[:n] & gripper_closed[:n]

    open_near = hands_near[:n] & ~closed_near
    if int(open_near.sum()) >= open_near_min_steps and not bool(closed_near.any()):
        return True

    if gripper_closed is None or len(gripper_closed) < n:
        return False

    eef_pos = _get_field(episode_group, "obs/right_eef_pos")
    if eef_pos is None:
        return False
    eef_pos = np.asarray(eef_pos).reshape(n, -1)[:, :3]
    eef_dz = np.diff(eef_pos[:, 2], prepend=eef_pos[0, 2])
    return bool((open_near & (eef_dz > 0.008)).any())


def classify_hand_motion_result(
    episode_group: h5py.Group,
    grasp_threshold: float = 0.20,
    spawn_ignore_steps: int = 5,
    open_near_min_steps: int = 3,
    knock_z_drop_m: float = 0.008,
    knock_xy_jerk_m: float = 0.008,
    knock_displacement_m: float = 0.003,
    knock_eef_speed_m: float = 0.003,
) -> str:
    """Classify how the hands interacted with the can (grasp vs knock)."""
    lift = _load_label_series(episode_group, "lift_latched")
    hands_near = _infer_hands_near(episode_group, grasp_threshold)
    gripper_closed = _infer_gripper_closed(episode_group)
    can_knocked = _infer_can_knocked(
        episode_group,
        hands_near,
        lift,
        gripper_closed,
        spawn_ignore_steps,
        knock_z_drop_m,
        knock_xy_jerk_m,
        knock_displacement_m,
        knock_eef_speed_m,
    )

    if hands_near is None or len(hands_near) == 0:
        return "no_hand_engagement"

    n = len(hands_near)
    if not bool(hands_near.any()):
        return "no_hand_engagement"

    closed_near = np.zeros(n, dtype=bool)
    if gripper_closed is not None and len(gripper_closed) >= n:
        closed_near = hands_near[:n] & gripper_closed[:n]

    lifted = bool(lift is not None and len(lift) >= n and lift[:n].any())

    if can_knocked is not None and len(can_knocked) >= n:
        knock_steps = np.where(can_knocked[:n])[0]
        if len(knock_steps) > 0:
            first_knock = int(knock_steps[0])
            closed_before = bool(closed_near[: first_knock + 1].any())
            lift_before = bool(lift is not None and len(lift) > first_knock and lift[: first_knock + 1].any())
            if not closed_before and not lift_before:
                return "knocked_can_without_grasp"

    if lifted and not bool(closed_near.any()):
        return "grasp_without_closing"

    if _infer_pick_without_close(episode_group, hands_near, gripper_closed, open_near_min_steps):
        return "grasp_without_closing"

    if bool(closed_near.any()) and not lifted:
        return "closed_without_lift"

    if _infer_spawn_knock(episode_group, spawn_ignore_steps, knock_z_drop_m) and not bool(closed_near.any()):
        return "knocked_can_without_grasp"

    return "valid_grasp_path"


def hand_motion_state_at_t(
    t: int,
    hands_near: np.ndarray | None,
    gripper_closed: np.ndarray | None,
    lift: np.ndarray | None,
    can_knocked: np.ndarray | None,
    pick_without_close: np.ndarray | None = None,
) -> str:
    """Per-step hand motion state for timestep labeling."""
    if lift is not None and t < len(lift) and bool(lift[t]):
        return "grasped"
    if can_knocked is not None and t < len(can_knocked) and bool(can_knocked[t]):
        return "knocked"
    if pick_without_close is not None and t < len(pick_without_close) and bool(pick_without_close[t]):
        return "approaching_open"
    if hands_near is not None and t < len(hands_near) and bool(hands_near[t]):
        if gripper_closed is not None and t < len(gripper_closed) and bool(gripper_closed[t]):
            return "closing"
        return "approaching_open"
    return "normal"


def classify_failure_type(failure_reasons: dict[str, bool] | None) -> str:
    """Classify failure type from the final termination-term snapshot."""
    if failure_reasons is None:
        return "unknown"

    # True = condition satisfied; failed terms are those that are False.
    failed_terms = frozenset(k for k, v in failure_reasons.items() if not v)
    if not failed_terms:
        return "unknown"

    # Check each taxonomy entry (most specific first)
    for pattern, ftype in sorted(FAILURE_TYPE_MAP.items(), key=lambda x: -len(x[0])):
        if pattern and failed_terms == pattern:
            return ftype

    # Fallback heuristics for partial matches
    if "object_dropping" in failed_terms:
        return "object_dropped_midway"
    if "time_out" in failed_terms:
        return "timeout"
    if "xy_in_tray" in failed_terms or "on_surface_z" in failed_terms:
        return "placement_failure"
    if "lift_latched" in failed_terms:
        return "lift_failed"
    if "hands_released" in failed_terms:
        return "grasp_held"
    return "unknown"


def _load_label_series(episode_group: h5py.Group, key: str) -> np.ndarray | None:
    """Load a per-step boolean label series from labels/{key}."""
    data = _get_field(episode_group, f"labels/{key}")
    if data is None:
        return None
    arr = np.asarray(data)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.dtype == bool or arr.dtype == np.bool_:
        return arr.astype(bool)
    return arr.astype(bool)


def infer_failure_step(episode_group: h5py.Group, failure_reasons: dict[str, bool] | None) -> int:
    """Estimate the first visible failure step from per-step labels or episode length."""
    n_samples = int(episode_group.attrs.get("num_samples", 0))
    if n_samples <= 0:
        return -1
    last_step = n_samples - 1

    lift = _load_label_series(episode_group, "lift_latched")
    released = _load_label_series(episode_group, "hands_released")
    in_xy = _load_label_series(episode_group, "xy_in_tray")
    in_z = _load_label_series(episode_group, "on_surface_z")
    obj_z = _get_field(episode_group, "labels/object_z")

    if lift is not None and len(lift) > 0:
        # Object dropped below table after being lifted
        if obj_z is not None and len(obj_z) == len(lift):
            obj_z = np.asarray(obj_z).reshape(-1)
            was_lifted = np.maximum.accumulate(lift.astype(int)).astype(bool)
            drop_mask = was_lifted & (obj_z < 0.95)
            drop_steps = np.where(drop_mask)[0]
            if len(drop_steps) > 0:
                return int(drop_steps[0])

        if in_xy is not None and len(in_xy) == len(lift):
            # Never reached tray — treat episode end as visible failure
            if bool(lift.any()) and not bool(in_xy.any()):
                return last_step

        latch_indices = np.where(lift)[0]
        if len(latch_indices) > 0 and released is not None and in_xy is not None:
            latch_start = int(latch_indices[0])
            # Premature release after pick, once the object was lifted
            for t in range(latch_start, min(len(released), len(in_xy))):
                if bool(released[t]) and not bool(in_xy[t]):
                    return t

    return last_step


def infer_root_cause_step(
    episode_group: h5py.Group,
    failure_step: int,
    failure_reasons: dict[str, bool] | None,
) -> int:
    """Estimate the earliest causal failure step (often before visible failure)."""
    if failure_step < 0:
        return -1

    lift = _load_label_series(episode_group, "lift_latched")
    released = _load_label_series(episode_group, "hands_released")
    in_xy = _load_label_series(episode_group, "xy_in_tray")
    hand_dist = _get_field(episode_group, "labels/min_hand_dist")

    # Grasp instability: hand distance spikes after lift latch
    if lift is not None and hand_dist is not None and len(lift) == len(hand_dist):
        hand_dist = np.asarray(hand_dist).reshape(-1)
        latched_idx = np.where(lift)[0]
        if len(latched_idx) > 0:
            latch_start = int(latched_idx[0])
            post = hand_dist[latch_start:failure_step + 1]
            if len(post) > 3:
                baseline = np.median(post[: max(3, len(post) // 4)])
                spike = np.where(post > baseline + 0.05)[0]
                if len(spike) > 0:
                    return latch_start + int(spike[0])

    # Released too early before reaching tray
    if released is not None and in_xy is not None and len(released) == len(in_xy):
        for t in range(min(failure_step, len(released))):
            if bool(released[t]) and not bool(in_xy[t]):
                return t

    # Lift never achieved
    if failure_reasons and not failure_reasons.get("lift_latched", True) and lift is not None:
        never = np.where(~lift)[0]
        if len(never) > 0:
            return int(never[len(never) // 2])

    return max(0, failure_step - 5)


def infer_recoverable(failure_reasons: dict[str, bool] | None, failure_type: str) -> bool:
    """Determine if the failure was potentially recoverable."""
    if failure_reasons is None:
        return False

    if failure_type in {"object_dropped", "object_dropped_midway", "timeout", "grasp_lost"}:
        return False

    # Still grasping — can retry placement
    if not failure_reasons.get("hands_released", True):
        return True

    # Lifted but not placed — may recover if still near tray
    if failure_reasons.get("lift_latched", False) and not failure_reasons.get("xy_in_tray", True):
        return True

    if failure_type in {"missed_tray", "placed_outside_tray", "placement_height_wrong", "lift_failed", "grasp_held"}:
        return True

    return False


def infer_near_failure_steps(
    episode_group: h5py.Group,
    failure_step: int,
    root_cause_step: int,
    window: int = 10,
) -> list[int]:
    """Identify near-failure steps before the visible failure."""
    if failure_step < 0:
        return []

    near_steps: set[int] = set()
    start = max(0, min(root_cause_step, failure_step) - window)
    end = min(failure_step, start + window * 2)

    lift = _load_label_series(episode_group, "lift_latched")
    released = _load_label_series(episode_group, "hands_released")
    in_xy = _load_label_series(episode_group, "xy_in_tray")
    hand_dist = _get_field(episode_group, "labels/min_hand_dist")

    if released is not None and in_xy is not None and len(released) == len(in_xy):
        for t in range(start, end):
            if bool(released[t]) and not bool(in_xy[t]):
                near_steps.add(t)

    if hand_dist is not None:
        hand_dist = np.asarray(hand_dist).reshape(-1)
        for t in range(start, end):
            if t >= len(hand_dist):
                break
            if hand_dist[t] < 0.18:
                near_steps.add(t)

    if lift is not None:
        for t in range(start, end):
            if t >= len(lift):
                break
            if not bool(lift[t]) and t > len(lift) // 3:
                near_steps.add(t)

    # Fallback: gripper motion proxy from joint positions
    if not near_steps:
        joint_data = _get_field(episode_group, "obs/joint_pos")
        if joint_data is not None and joint_data.ndim == 2 and joint_data.shape[1] > 30:
            gripper_pos = joint_data[:, GRIPPER_INDICES]
            for t in range(start, end):
                if t >= len(gripper_pos):
                    break
                if np.any(np.abs(gripper_pos[t]) > 0.05):
                    near_steps.add(t)

    return sorted(near_steps)


def _get_field(episode_group: h5py.Group, key: str):
    """Extract a field from an episode HDF5 group.

    Handles both tensor datasets and nested dict groups (e.g., obs/joint_pos
    stored as obs -> joint_pos group with element datasets).
    """
    parts = key.split("/")
    current = episode_group

    for part in parts[:-1]:
        if part in current:
            current = current[part]
        else:
            return None

    last = parts[-1]
    if last not in current:
        return None

    leaf = current[last]
    if isinstance(leaf, h5py.Dataset):
        return leaf[:]
    elif isinstance(leaf, h5py.Group):
        # Nested dict — return the group for further processing
        return leaf
    return None


# ---------------------------------------------------------------------------
# Timestep-level analysis
# ---------------------------------------------------------------------------

def analyze_timestep_state(
    t: int,
    failure_step: int,
    root_cause_step: int,
    near_failure_steps: set[int],
    labels: dict[str, np.ndarray | None],
) -> str:
    """Classify timestep state for a single frame."""
    if t >= failure_step:
        return "irreversible_failure"
    if t in near_failure_steps or (root_cause_step >= 0 and t >= root_cause_step):
        return "degraded"

    lift = labels.get("lift_latched")
    released = labels.get("hands_released")
    in_xy = labels.get("xy_in_tray")
    hand_dist = labels.get("min_hand_dist")

    if hand_dist is not None and t < len(hand_dist) and hand_dist[t] < 0.12:
        return "unsafe"
    if released is not None and in_xy is not None and t < len(released):
        if bool(released[t]) and not bool(in_xy[t]):
            return "degraded"
    if lift is not None and t < len(lift) and not bool(lift[t]) and t > max(1, failure_step // 3):
        return "degraded"
    return "normal"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze failed episodes and produce labeled failure dataset."
    )
    parser.add_argument("--input_file", type=str, required=True, help="Path to *_failed.hdf5 file")
    parser.add_argument("--output_file", type=str, default="failure_labeled.hdf5", help="Output HDF5 path")
    parser.add_argument("--env_name", type=str, default="a2_pickplace", help="Environment name for metadata")
    parser.add_argument("--min_episodes", type=int, default=1, help="Minimum episodes to process")
    parser.add_argument("--max_episodes", type=int, default=None, help="Maximum episodes to process (None=all)")
    parser.add_argument("--include_images", action="store_true", help="Include downsampled images in output")
    parser.add_argument("--include_timesteps", action="store_true", help="Include per-timestep labels")
    parser.add_argument("--grasp_threshold", type=float, default=0.20, help="Hand-near-can distance threshold (m)")
    parser.add_argument("--spawn_ignore_steps", type=int, default=5, help="Steps to ignore after reset for knock detection")
    parser.add_argument("--open_near_min_steps", type=int, default=3, help="Min open-near steps for grasp_without_closing")
    parser.add_argument("--save_joint_names", action="store_true", help="Save joint names JSON alongside output")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"[capture_failure] Reading {input_path}")
    with h5py.File(input_path, "r") as f:
        demo_keys = sorted(
            [k for k in f["data"].keys() if k.startswith("demo_")],
            key=lambda k: int(k.split("_")[-1]),
        )

    if not demo_keys:
        print(f"[capture_failure] No episodes found in {input_path}")
        return

    max_ep = args.max_episodes or len(demo_keys)
    print(f"[capture_failure] Found {len(demo_keys)} failed episodes, processing up to {max_ep}")

    # Environment metadata
    env_args = {}
    with h5py.File(input_path, "r") as f:
        if "data" in f and "env_args" in f["data"].attrs:
            try:
                env_args = json.loads(f["data"].attrs["env_args"])
            except (json.JSONDecodeError, KeyError):
                pass

    # Collect trajectory-level labels
    traj_metadata: dict[str, list] = defaultdict(list)
    episode_states: dict[str, dict] = {}  # For timestep output

    with h5py.File(input_path, "r") as f:
        for i, key in enumerate(_tqdm.tqdm(demo_keys[:max_ep], desc="Analyzing failures")):
            ep_group = f[f"data/{key}"]
            failure_reasons_raw = None
            if "failure_reasons" in ep_group:
                fr_group = ep_group["failure_reasons"]
                failure_reasons_raw = {
                    name: bool(fr_group.attrs[name]) for name in fr_group.attrs.keys()
                }

            n_samples = int(ep_group.attrs.get("num_samples", 0))
            seed = int(ep_group.attrs.get("seed", -1)) if "seed" in ep_group.attrs else -1
            success_val = ep_group.attrs.get("success", -1)
            is_success = bool(success_val) if success_val != -1 else False

            # Skip successful episodes (shouldn't be in _failed.hdf5 but just in case)
            if is_success:
                continue

            failure_step = infer_failure_step(ep_group, failure_reasons_raw)
            failure_type = classify_failure_type(failure_reasons_raw)
            root_cause_step = infer_root_cause_step(ep_group, failure_step, failure_reasons_raw)
            recoverable = infer_recoverable(failure_reasons_raw, failure_type)
            severity = SEVERITY_MAP.get(failure_type, "medium")
            near_failure_steps = infer_near_failure_steps(ep_group, failure_step, root_cause_step)
            hand_motion_result = classify_hand_motion_result(
                ep_group,
                args.grasp_threshold,
                args.spawn_ignore_steps,
                args.open_near_min_steps,
            )

            hands_near = _infer_hands_near(ep_group, args.grasp_threshold)
            gripper_closed = _infer_gripper_closed(ep_group)
            lift_series = _load_label_series(ep_group, "lift_latched")
            can_knocked = _infer_can_knocked(
                ep_group, hands_near, lift_series, gripper_closed, args.spawn_ignore_steps,
            )

            episode_states[key] = {
                "failure_step": failure_step,
                "root_cause_step": root_cause_step,
                "near_failure_steps": set(near_failure_steps),
                "hand_motion_result": hand_motion_result,
                "labels": {
                    "lift_latched": lift_series,
                    "hands_released": _load_label_series(ep_group, "hands_released"),
                    "xy_in_tray": _load_label_series(ep_group, "xy_in_tray"),
                    "hands_near_can": hands_near,
                    "gripper_closed": gripper_closed,
                    "can_knocked_now": can_knocked,
                    "pick_without_close_now": _load_label_series(ep_group, "pick_without_close_now"),
                    "min_hand_dist": (
                        np.asarray(_get_field(ep_group, "labels/min_hand_dist")).reshape(-1)
                        if _get_field(ep_group, "labels/min_hand_dist") is not None
                        else None
                    ),
                },
            }

            # Collect trajectory-level metadata
            traj_metadata["trajectory_id"].append(key)
            traj_metadata["outcome"].append("failure")
            traj_metadata["failure_type"].append(failure_type)
            traj_metadata["failure_step"].append(failure_step)
            traj_metadata["root_cause_step"].append(root_cause_step)
            traj_metadata["visible_failure_step"].append(failure_step)
            traj_metadata["recoverable"].append(int(recoverable))
            traj_metadata["severity"].append(severity)
            traj_metadata["near_failure_count"].append(len(near_failure_steps))
            traj_metadata["near_failure_steps"].append(json.dumps(near_failure_steps))
            traj_metadata["hand_motion_result"].append(hand_motion_result)
            traj_metadata["seed"].append(seed)
            traj_metadata["num_steps"].append(n_samples)

            # Raw failure reasons as a compact dict of failed terms
            if failure_reasons_raw:
                failed_terms = {k: v for k, v in failure_reasons_raw.items() if not v}
                traj_metadata["failed_terms"].append(json.dumps(failed_terms))
            else:
                traj_metadata["failed_terms"].append("{}")

    if not traj_metadata["trajectory_id"]:
        print("[capture_failure] No failed episodes after filtering.")
        return

    # Write output
    out_path = Path(args.output_file)
    print(f"[capture_failure] Writing labeled failure dataset to {out_path}")

    with h5py.File(out_path, "w") as out:
        # --- Failed episodes group ---
        failed_grp = out.create_group("failed")
        for i, key in enumerate(traj_metadata["trajectory_id"]):
            src_key = key
            with h5py.File(input_path, "r") as f:
                src_grp = f[f"data/{src_key}"]
                ep_grp = failed_grp.create_group(key)

                # Copy all data from source (except failure_reasons — we rewrite)
                for k in src_grp.keys():
                    src_grp.copy(k, ep_grp)

                # Add trajectory-level labels as attributes
                ep_grp.attrs["failure_type"] = traj_metadata["failure_type"][i]
                ep_grp.attrs["severity"] = traj_metadata["severity"][i]
                ep_grp.attrs["recoverable"] = traj_metadata["recoverable"][i]
                ep_grp.attrs["failure_step"] = traj_metadata["failure_step"][i]
                ep_grp.attrs["root_cause_step"] = traj_metadata["root_cause_step"][i]
                ep_grp.attrs["near_failure_count"] = traj_metadata["near_failure_count"][i]
                ep_grp.attrs["hand_motion_result"] = traj_metadata["hand_motion_result"][i]
                ep_grp.attrs["seed"] = traj_metadata["seed"][i]
                ep_grp.attrs["num_steps"] = traj_metadata["num_steps"][i]

                # Near-failure steps
                near_str = traj_metadata["near_failure_steps"][i]
                if near_str:
                    ep_grp.attrs["near_failure_steps"] = near_str

                # Failed terms (the specific terms that failed)
                ep_grp.attrs["failed_terms"] = traj_metadata["failed_terms"][i]

                # --- Timestep-level labels ---
                if args.include_timesteps:
                    st = episode_states.get(src_key, {})
                    fs = st.get("failure_step", -1)
                    rcs = st.get("root_cause_step", -1)
                    nfs = st.get("near_failure_steps", set())
                    labels = st.get("labels", {})
                    n_samples = int(ep_grp.attrs.get("num_steps", 0))
                    T = n_samples if n_samples > 0 else fs + 1
                    if T <= 0 and "actions" in src_grp:
                        T = len(src_grp["actions"])
                    if T > 0:
                        state_labels = np.array([
                            analyze_timestep_state(t, fs, rcs, nfs, labels)
                            for t in range(T)
                        ], dtype=object)

                        # Map state strings to numeric codes
                        STATE_MAP = {
                            "normal": 0, "degraded": 1, "unsafe": 2,
                            "irreversible_failure": 3, "recovery_attempt": 4,
                        }
                        state_codes = np.array([STATE_MAP.get(s, 0) for s in state_labels], dtype=np.int8)
                        ep_grp.create_dataset("timestep_state", data=state_codes)
                        ep_grp.create_dataset("timestep_state_label", data=np.array([s.encode("utf-8") for s in state_labels], dtype="S24"))

                        # Near-failure flags
                        is_near = np.zeros(T, dtype=bool)
                        for ns in nfs:
                            if 0 <= ns < T:
                                is_near[ns] = True
                        ep_grp.create_dataset("is_near_failure", data=is_near)

                        # Root-cause window: steps within 5 before failure_step
                        root_start = max(0, fs - 5)
                        root_end = min(fs, T)
                        is_root = np.zeros(T, dtype=bool)
                        is_root[root_start:root_end] = True
                        ep_grp.create_dataset("is_root_cause_window", data=is_root)

                        hand_motion_labels = np.array([
                            hand_motion_state_at_t(
                                t,
                                labels.get("hands_near_can"),
                                labels.get("gripper_closed"),
                                labels.get("lift_latched"),
                                labels.get("can_knocked_now"),
                                labels.get("pick_without_close_now"),
                            )
                            for t in range(T)
                        ], dtype=object)
                        HAND_MOTION_STATE_MAP = {
                            "normal": 0,
                            "approaching_open": 1,
                            "closing": 2,
                            "grasped": 3,
                            "knocked": 4,
                        }
                        ep_grp.create_dataset(
                            "hand_motion_state",
                            data=np.array([HAND_MOTION_STATE_MAP.get(s, 0) for s in hand_motion_labels], dtype=np.int8),
                        )
                        ep_grp.create_dataset(
                            "hand_motion_state_label",
                            data=np.array([s.encode("utf-8") for s in hand_motion_labels], dtype="S16"),
                        )

        # --- Trajectory-level metadata table ---
        traj_meta = out.create_group("traj_meta")
        traj_meta.create_dataset("trajectory_id", data=np.array(traj_metadata["trajectory_id"], dtype="S16"))
        traj_meta.create_dataset("outcome", data=np.array(traj_metadata["outcome"], dtype="S8"))
        traj_meta.create_dataset("failure_type", data=np.array(traj_metadata["failure_type"], dtype="S20"))
        traj_meta.create_dataset("severity", data=np.array(traj_metadata["severity"], dtype="S10"))
        traj_meta.create_dataset("failed_terms", data=np.array(traj_metadata["failed_terms"], dtype="S128"))
        traj_meta.create_dataset("failure_step", data=np.array(traj_metadata["failure_step"], dtype=np.int32))
        traj_meta.create_dataset("root_cause_step", data=np.array(traj_metadata["root_cause_step"], dtype=np.int32))
        traj_meta.create_dataset("recoverable", data=np.array(traj_metadata["recoverable"], dtype=bool))
        traj_meta.create_dataset("near_failure_count", data=np.array(traj_metadata["near_failure_count"], dtype=np.int32))
        traj_meta.create_dataset(
            "hand_motion_result",
            data=np.array(traj_metadata["hand_motion_result"], dtype="S28"),
        )
        traj_meta.create_dataset("seed", data=np.array(traj_metadata["seed"], dtype=np.int32))
        traj_meta.create_dataset("num_steps", data=np.array(traj_metadata["num_steps"], dtype=np.int32))

        # --- Failure catalog summary ---
        catalog = out.create_group("failure_catalog")
        counts = defaultdict(int)
        for ft in traj_metadata["failure_type"]:
            counts[ft] += 1
        for ft, cnt in sorted(counts.items()):
            catalog.attrs[ft] = cnt

        sev_counts = defaultdict(int)
        for sv in traj_metadata["severity"]:
            sev_counts[sv] += 1
        for sv, cnt in sorted(sev_counts.items()):
            catalog.attrs[f"severity:{sv}"] = cnt

        hand_motion_counts = defaultdict(int)
        for hm in traj_metadata["hand_motion_result"]:
            hand_motion_counts[hm] += 1
        for hm, cnt in sorted(hand_motion_counts.items()):
            catalog.attrs[f"hand_motion:{hm}"] = cnt

        n_recoverable = sum(traj_metadata["recoverable"])
        catalog.attrs["n_episodes"] = len(traj_metadata["trajectory_id"])
        catalog.attrs["n_recoverable"] = int(n_recoverable)
        catalog.attrs["recoverable_ratio"] = float(n_recoverable) / max(len(traj_metadata["trajectory_id"]), 1)

        # --- Environmental context ---
        env_grp = out.create_group("env_args")
        for k, v in env_args.items():
            env_grp.attrs[k] = str(v) if not isinstance(v, (int, float, bool)) else v

    # Print summary
    print(f"\n[capture_failure] {'='*50}")
    print(f"[capture_failure] Failure Analysis Summary ({len(traj_metadata['trajectory_id'])} episodes)")
    print(f"[capture_failure] {'='*50}")
    print(f"  Recovery rate: {n_recoverable}/{len(traj_metadata['trajectory_id'])} "
          f"({float(n_recoverable)/len(traj_metadata['trajectory_id'])*100:.1f}%)")
    print(f"\n  Failure type distribution:")
    for ft, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"    {ft}: {cnt}")
    print(f"\n  Severity distribution:")
    for sv, cnt in sorted(sev_counts.items(), key=lambda x: {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(x[0], 4)):
        print(f"    {sv}: {cnt}")
    print(f"\n  Hand motion result distribution:")
    for hm, cnt in sorted(hand_motion_counts.items(), key=lambda x: -x[1]):
        print(f"    {hm}: {cnt}")
    print(f"[capture_failure] Output: {out_path}")

    # Save joint names JSON for downstream converters
    if args.save_joint_names:
        jn_path = out_path.with_suffix(".json")
        with open(jn_path, "w") as f:
            json.dump({"joint_names": A2_JOINT_NAMES}, f, indent=2)
        print(f"[capture_failure] Joint names saved: {jn_path}")


if __name__ == "__main__":
    main()
