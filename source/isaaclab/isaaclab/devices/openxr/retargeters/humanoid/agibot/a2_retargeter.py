# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OpenXR hand tracking → A2 humanoid retargeter (Isaac Lab adapter).

Mirrors the structure of gr1t2_retargeter.py. Output tensor layout per frame:
    [left_wrist_pose (7), right_wrist_pose (7), hand_joint_angles (len(hand_joint_names))]

Hand joint angles come from A2DexRetargeting.compute_left/compute_right() and are
placed at the indices determined by `hand_joint_names`. The config's hand_joint_names
is used BOTH to size the output and to determine which slots receive values; joints
not produced by the retargeter (mimic children) stay at 0 — the env-side controller
must set them from the URDF mimic multipliers each step.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

import isaaclab.sim as sim_utils
from isaaclab.devices.device_base import DeviceBase
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

with contextlib.suppress(Exception):
    from .a2_dex_retargeting_utils import A2DexRetargeting


# A2.urdf defines a fixed rotation from arm_link07 to the hand anatomical frame:
#   <joint name="left_arm_joint07_fixed"  rpy="1.5708  0 -1.5708" />
#   <joint name="right_arm_joint07_fixed" rpy="-1.5708 0 -1.5708" />
# merge_fixed_joints: true collapses `left_hand`/`right_hand` into arm_link07, so
# Pink IK's target frame is arm_link07. To make the hand visually track the user's
# hand, we pre-multiply the user's world-frame wrist rotation by the *inverse* of
# the fixed-joint rotation: target_link07_rot = user_hand_rot × R_fixed_joint.inv()
# which yields R_link07 s.t. (R_link07 × R_fixed_joint) == user_hand_rot.
# Wrist pre-compensation rpy per side — empirically tuned against A2's rendered
# hand pose during VR hand-tracking (2026-04-21). Both hands need the *same* pure
# +π/2 X-roll; despite A2.urdf's differing fixed-joint rpy per side, after USD
# conversion + merge_fixed_joints the effective control frame ends up symmetric.
_DEFAULT_LEFT_RPY = (1.5707963, 0.0, 0.0)
_DEFAULT_RIGHT_RPY = (1.5707963, 0.0, 0.0)


def _compose_openxr_wrist(
    openxr_wrist: np.ndarray,
    fixed_rot: R,
    shoulder_pos: np.ndarray | None = None,
    max_reach: float = 0.7,
    pos_scale: float = 1.0,
) -> np.ndarray:
    """OpenXR wrist pose → Pink IK target for arm_link07 with:
      1) orientation pre-compensated (fixed_rot.inv applied).
      2) Optional position scaling around the shoulder (pos_scale, 1.0 = pass-through).
      3) Clipping to a reachable sphere of radius max_reach centered at shoulder_pos,
         so Pink IK always has a valid target even if the user reaches outside.
    """
    pos = openxr_wrist[:3].astype(np.float32)

    if shoulder_pos is not None:
        offset = pos - shoulder_pos
        # Scale the user's hand offset from their shoulder (compresses motion range).
        offset = offset * pos_scale
        # Clamp to reachable sphere.
        dist = float(np.linalg.norm(offset))
        if dist > max_reach:
            offset = offset * (max_reach / dist)
        pos = shoulder_pos + offset

    # OpenXR gives quaternion as (w, x, y, z); scipy takes (x, y, z, w).
    w, x, y, z = openxr_wrist[3], openxr_wrist[4], openxr_wrist[5], openxr_wrist[6]
    user_rot = R.from_quat([x, y, z, w])
    target_rot = user_rot * fixed_rot.inv()
    qx, qy, qz, qw = target_rot.as_quat()
    return np.array([pos[0], pos[1], pos[2], qw, qx, qy, qz], dtype=np.float32)


class A2Retargeter(RetargeterBase):
    """Retargets OpenXR hand tracking → A2 s6_hand joint angles + wrist poses."""

    def __init__(self, cfg: A2RetargeterCfg):
        super().__init__(cfg)
        self._cfg = cfg
        self._hand_joint_names = cfg.hand_joint_names
        self._hands_controller = A2DexRetargeting(self._hand_joint_names)

        self._enable_visualization = cfg.enable_visualization
        self._num_open_xr_hand_joints = cfg.num_open_xr_hand_joints
        self._sim_device = cfg.sim_device

        # Per-side pre-compensation rotations (build once per env). Set in env cfg
        # via A2RetargeterCfg.left_fixed_rpy / right_fixed_rpy; a2_retargeter logs
        # both values on the first retarget() call to aid tuning.
        self._left_fixed_rot = R.from_euler(cfg.euler_convention, cfg.left_fixed_rpy)
        self._right_fixed_rot = R.from_euler(cfg.euler_convention, cfg.right_fixed_rpy)
        self._logged_rpy = False
        self._debug_every = cfg.debug_every  # 0 disables; N>0 prints every N frames
        self._frame_count = 0

        # Auto-calibration: when both wrists stay still (palm-down expected) for
        # `auto_calibrate_seconds`, capture the user's current rotations as fixed_rot.
        # 0 disables (manual cfg values used).
        self._auto_calibrate_seconds = float(cfg.auto_calibrate_seconds)
        self._auto_calibrate_threshold_rad = float(np.deg2rad(cfg.auto_calibrate_threshold_deg))
        self._calibrated = self._auto_calibrate_seconds <= 0.0
        self._stable_anchor_left: R | None = None
        self._stable_anchor_right: R | None = None
        self._stable_start_time: float | None = None
        self._last_countdown_print: float = 0.0

        if self._enable_visualization:
            marker_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/markers",
                markers={
                    "joint": sim_utils.SphereCfg(
                        radius=0.005,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.3)),
                    ),
                },
            )
            self._markers = VisualizationMarkers(marker_cfg)

    def _maybe_auto_calibrate(self, left_wrist: np.ndarray, right_wrist: np.ndarray) -> None:
        """Auto-capture fixed_rot when both wrists stay still for `auto_calibrate_seconds`.

        Stability = current quat within `auto_calibrate_threshold_deg` of the anchor
        quat that began the current stable window. Any larger drift resets the anchor.
        Once captured, the new rotations replace `_left_fixed_rot` / `_right_fixed_rot`
        and the captured rpy is printed so it can be pasted back into cfg if desired.
        """
        lw, rw = np.asarray(left_wrist), np.asarray(right_wrist)
        # OpenXR wrist layout: [px, py, pz, qw, qx, qy, qz]; scipy wants (x, y, z, w).
        lq = R.from_quat([lw[4], lw[5], lw[6], lw[3]])
        rq = R.from_quat([rw[4], rw[5], rw[6], rw[3]])

        now = time.monotonic()
        if self._stable_anchor_left is None:
            self._stable_anchor_left = lq
            self._stable_anchor_right = rq
            self._stable_start_time = now
            print(
                f"[A2Retargeter][AUTO_CAL] Watching for stillness — keep both palms "
                f"down and steady for {self._auto_calibrate_seconds:.1f}s..."
            )
            return

        rel_l = float((self._stable_anchor_left.inv() * lq).magnitude())
        rel_r = float((self._stable_anchor_right.inv() * rq).magnitude())
        if rel_l > self._auto_calibrate_threshold_rad or rel_r > self._auto_calibrate_threshold_rad:
            self._stable_anchor_left = lq
            self._stable_anchor_right = rq
            self._stable_start_time = now
            return

        elapsed = now - self._stable_start_time
        if elapsed >= self._auto_calibrate_seconds:
            self._left_fixed_rot = lq
            self._right_fixed_rot = rq
            self._calibrated = True
            l_rpy = lq.as_euler("xyz")
            r_rpy = rq.as_euler("xyz")
            l_deg = np.round(np.degrees(l_rpy), 1).tolist()
            r_deg = np.round(np.degrees(r_rpy), 1).tolist()
            # Each line independently prefixed so grep/filters surface them all.
            print(f"[A2Retargeter][AUTO_CAL] CAPTURED after {elapsed:.1f}s stable.")
            print(f"[A2Retargeter][AUTO_CAL] left_rpy°  = {l_deg}")
            print(f"[A2Retargeter][AUTO_CAL] right_rpy° = {r_deg}")
            print(f"[A2Retargeter][AUTO_CAL] Paste into cfg to persist:")
            print(
                f"[A2Retargeter][AUTO_CAL]     "
                f"left_fixed_rpy=({l_rpy[0]:.4f}, {l_rpy[1]:.4f}, {l_rpy[2]:.4f}),"
            )
            print(
                f"[A2Retargeter][AUTO_CAL]     "
                f"right_fixed_rpy=({r_rpy[0]:.4f}, {r_rpy[1]:.4f}, {r_rpy[2]:.4f}),"
            )
            return

        # Periodic countdown (~once per second).
        if now - self._last_countdown_print >= 1.0:
            self._last_countdown_print = now
            print(
                f"[A2Retargeter][AUTO_CAL] Stable {elapsed:.1f}/"
                f"{self._auto_calibrate_seconds:.1f}s "
                f"(drift L={np.degrees(rel_l):.1f}° R={np.degrees(rel_r):.1f}°)"
            )

    def retarget(self, data: dict) -> torch.Tensor:
        left_hand_poses = data[DeviceBase.TrackingTarget.HAND_LEFT]
        right_hand_poses = data[DeviceBase.TrackingTarget.HAND_RIGHT]

        left_wrist = left_hand_poses.get("wrist")
        right_wrist = right_hand_poses.get("wrist")

        if not self._calibrated and left_wrist is not None and right_wrist is not None:
            self._maybe_auto_calibrate(left_wrist, right_wrist)

        if self._enable_visualization:
            joints_position = np.zeros((self._num_open_xr_hand_joints, 3))
            joints_position[::2] = np.array([pose[:3] for pose in left_hand_poses.values()])
            joints_position[1::2] = np.array([pose[:3] for pose in right_hand_poses.values()])
            self._markers.visualize(translations=torch.tensor(joints_position, device=self._sim_device))

        # Place retargeter outputs at the correct slots of hand_joint_names via
        # name lookup. Pair (name, value) before indexing so any name missing
        # from hand_joint_names is safely skipped without scrambling positions.
        retargeted_hand_joints = np.zeros(len(self._hand_joint_names), dtype=np.float32)

        left_hands_pos = self._hands_controller.compute_left(left_hand_poses)
        for name, val in zip(self._hands_controller.get_left_joint_names(), left_hands_pos):
            if name in self._hand_joint_names:
                retargeted_hand_joints[self._hand_joint_names.index(name)] = val

        right_hands_pos = self._hands_controller.compute_right(right_hand_poses)
        for name, val in zip(self._hands_controller.get_right_joint_names(), right_hands_pos):
            if name in self._hand_joint_names:
                retargeted_hand_joints[self._hand_joint_names.index(name)] = val

        if not self._logged_rpy:
            print(
                f"[A2Retargeter] wrist pre-compensation rpy (left, right)="
                f"{self._left_fixed_rot.as_euler('xyz').tolist()}, "
                f"{self._right_fixed_rot.as_euler('xyz').tolist()}"
            )
            self._logged_rpy = True
        left_target = _compose_openxr_wrist(
            np.asarray(left_wrist), self._left_fixed_rot,
            shoulder_pos=self._cfg.left_shoulder_pos,
            max_reach=self._cfg.max_reach, pos_scale=self._cfg.pos_scale,
        )
        right_target = _compose_openxr_wrist(
            np.asarray(right_wrist), self._right_fixed_rot,
            shoulder_pos=self._cfg.right_shoulder_pos,
            max_reach=self._cfg.max_reach, pos_scale=self._cfg.pos_scale,
        )

        self._frame_count += 1

        left_wrist_tensor = torch.tensor(left_target, dtype=torch.float32, device=self._sim_device)
        right_wrist_tensor = torch.tensor(right_target, dtype=torch.float32, device=self._sim_device)
        hand_joints_tensor = torch.tensor(
            retargeted_hand_joints, dtype=torch.float32, device=self._sim_device
        )
        return torch.cat([left_wrist_tensor, right_wrist_tensor, hand_joints_tensor])

    def get_requirements(self) -> list[RetargeterBase.Requirement]:
        return [RetargeterBase.Requirement.HAND_TRACKING]


@dataclass
class A2RetargeterCfg(RetargeterCfg):
    """Config for A2 humanoid hand retargeter."""

    enable_visualization: bool = False
    num_open_xr_hand_joints: int = 100
    hand_joint_names: list[str] | None = None
    # Per-side wrist pre-compensation rpy. Applied as target = user_rot × R(rpy).inv().
    # Tune empirically: see comments near _DEFAULT_*_RPY for guidance.
    left_fixed_rpy: tuple[float, float, float] = _DEFAULT_LEFT_RPY
    right_fixed_rpy: tuple[float, float, float] = _DEFAULT_RIGHT_RPY
    # scipy euler convention string: "xyz" = extrinsic (world-fixed), "XYZ" = intrinsic (body-fixed).
    euler_convention: str = "xyz"
    # Print wrist targets + driven hand-joint values every N frames. 0 = off.
    # Set to ~30 (≈1 Hz at 30 Hz) to see live values without flooding stdout.
    debug_every: int = 0

    # Auto-calibration: when both wrists stay within `auto_calibrate_threshold_deg`
    # of their anchor for `auto_calibrate_seconds`, capture current rotations as
    # fixed_rot. 0 disables (manual cfg values used as-is).
    auto_calibrate_seconds: float = 0.0
    auto_calibrate_threshold_deg: float = 5.0

    # Shoulder world-frame positions (for reach clipping). Robot at pos=(0,0,0.93),
    # rot=(0.7071,0,0,0.7071) means body +x (forward) → world +y, body -y (right)
    # → world +x. Arm shoulders ≈ (±0.2, 0, 1.1) in world. Override per-scene.
    left_shoulder_pos: tuple[float, float, float] = (-0.2, 0.0, 1.1)
    right_shoulder_pos: tuple[float, float, float] = (0.2, 0.0, 1.1)
    # Clip user's wrist target to this radius around the respective shoulder.
    max_reach: float = 0.7
    # Scale the user's wrist offset from shoulder (1.0 = pass-through; 0.6 compresses
    # the user's 1m physical range into a 0.6m robot range — useful if the user's
    # natural motion overshoots the robot's workspace).
    pos_scale: float = 1.0

    retargeter_type: type[RetargeterBase] = A2Retargeter
