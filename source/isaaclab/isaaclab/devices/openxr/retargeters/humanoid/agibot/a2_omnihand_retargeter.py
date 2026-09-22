# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OpenXR hand tracking → A2 + OmniHand T2 retargeter (Isaac Lab adapter).

Analogue of ``a2_retargeter.py`` but drives the 10-DoF OmniHand T2 on each
wrist instead of the stock A2 s6_hand. Output layout is identical:
    [left_wrist_pose (7), right_wrist_pose (7), hand_joint_angles (len(hand_joint_names))]
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

from .a2_retargeter import _compose_openxr_wrist  # reuse the arm-wrist helper

with contextlib.suppress(Exception):
    from .a2_omnihand_dex_retargeting_utils import A2OmniHandDexRetargeting


# Wrist pre-compensation — OmniHand attaches with the same fixed-joint rpy the
# stock A2 hand uses (left_arm_joint07_fixed rpy="1.5708 0 -1.5708"; right rpy
# inverted on X). After USD conversion + merge_fixed_joints, both sides respond
# symmetrically to a +π/2 X-roll, same as the s6_hand A2 setup.
_DEFAULT_LEFT_RPY = (1.5707963, 0.0, 0.0)
_DEFAULT_RIGHT_RPY = (1.5707963, 0.0, 0.0)


class A2OmniHandRetargeter(RetargeterBase):
    """Retargets OpenXR hand tracking → A2 wrist poses + OmniHand joint angles."""

    def __init__(self, cfg: A2OmniHandRetargeterCfg):
        super().__init__(cfg)
        self._cfg = cfg
        self._hand_joint_names = cfg.hand_joint_names
        self._hands_controller = A2OmniHandDexRetargeting(self._hand_joint_names)

        self._enable_visualization = cfg.enable_visualization
        self._num_open_xr_hand_joints = cfg.num_open_xr_hand_joints
        self._sim_device = cfg.sim_device

        self._left_fixed_rot = R.from_euler(cfg.euler_convention, cfg.left_fixed_rpy)
        self._right_fixed_rot = R.from_euler(cfg.euler_convention, cfg.right_fixed_rpy)
        self._logged_rpy = False
        self._debug_every = cfg.debug_every
        self._frame_count = 0

        # Auto-calibration: when both wrists stay still for `auto_calibrate_seconds`,
        # capture the user's current rotations as fixed_rot (mirrors a2_retargeter).
        self._auto_calibrate_seconds = float(cfg.auto_calibrate_seconds)
        self._auto_calibrate_threshold_rad = float(np.deg2rad(cfg.auto_calibrate_threshold_deg))
        self._calibrated = self._auto_calibrate_seconds <= 0.0
        self._stable_anchor_left: R | None = None
        self._stable_anchor_right: R | None = None
        self._stable_start_time: float | None = None
        self._last_countdown_print: float = 0.0
        # Freeze detector: real hand tracking is noisy at the sub-degree level.
        # Many consecutive zero-drift samples ⇒ tracking dropped and the runtime
        # is replaying the last quat; abort cal to avoid latching stale values.
        self._frozen_streak: int = 0
        self._frozen_max_streak: int = 30  # ~1 s at 30 Hz retargeter tick

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
        """Capture fixed_rot when both wrists stay within drift threshold for N seconds."""
        lw, rw = np.asarray(left_wrist), np.asarray(right_wrist)
        # OpenXR wrist layout: [px, py, pz, qw, qx, qy, qz]; scipy wants (x, y, z, w).
        # Skip frames where the OpenXR runtime is returning identity quat + zero
        # position — that's the signal that hand tracking hasn't actually started
        # yet (headset not on, hands not visible). Without this guard the cal can
        # lock onto the zero-state before the user's hands ever stream real data.
        def _is_uninitialized(arr: np.ndarray) -> bool:
            pos = arr[:3]
            qw, qx, qy, qz = arr[3], arr[4], arr[5], arr[6]
            return (
                bool(np.all(np.abs(pos) < 1e-6))
                and abs(qw - 1.0) < 1e-6 and abs(qx) < 1e-6
                and abs(qy) < 1e-6 and abs(qz) < 1e-6
            )
        if _is_uninitialized(lw) or _is_uninitialized(rw):
            now = time.monotonic()
            if now - self._last_countdown_print >= 2.0:
                self._last_countdown_print = now
                print(
                    "[A2OmniHandRetargeter][AUTO_CAL] Waiting for hand tracking — "
                    "headset must be on and both hands visible to the cameras."
                )
            return
        lq = R.from_quat([lw[4], lw[5], lw[6], lw[3]])
        rq = R.from_quat([rw[4], rw[5], rw[6], rw[3]])

        now = time.monotonic()
        if self._stable_anchor_left is None:
            self._stable_anchor_left = lq
            self._stable_anchor_right = rq
            self._stable_start_time = now
            print(
                f"[A2OmniHandRetargeter][AUTO_CAL] Watching for stillness — keep both palms "
                f"down and steady for {self._auto_calibrate_seconds:.1f}s..."
            )
            return

        rel_l = float((self._stable_anchor_left.inv() * lq).magnitude())
        rel_r = float((self._stable_anchor_right.inv() * rq).magnitude())
        # Freeze detector: real hand tracking has sub-degree per-sample noise.
        # Multiple consecutive samples with drift==0.0° on either hand means the
        # OpenXR runtime has stopped streaming and is repeating the last quat.
        # Capturing during a freeze locks in stale values.
        is_frozen = (rel_l < 1e-9) or (rel_r < 1e-9)
        if is_frozen:
            self._frozen_streak += 1
        else:
            self._frozen_streak = 0
        if self._frozen_streak >= self._frozen_max_streak:
            if now - self._last_countdown_print >= 2.0:
                self._last_countdown_print = now
                print(
                    f"[A2OmniHandRetargeter][AUTO_CAL] Tracking appears frozen "
                    f"({self._frozen_streak} consecutive zero-drift samples) — "
                    f"pausing cal. Wave hands to resume."
                )
            # Keep the anchor so the timer is preserved across a brief freeze;
            # but DO NOT advance elapsed when we're frozen. Just return.
            return
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
            print(f"[A2OmniHandRetargeter][AUTO_CAL] CAPTURED after {elapsed:.1f}s stable.")
            print(f"[A2OmniHandRetargeter][AUTO_CAL] left_rpy°  = {l_deg}")
            print(f"[A2OmniHandRetargeter][AUTO_CAL] right_rpy° = {r_deg}")
            print(f"[A2OmniHandRetargeter][AUTO_CAL] Paste into cfg to persist:")
            print(
                f"[A2OmniHandRetargeter][AUTO_CAL]     "
                f"left_fixed_rpy=({l_rpy[0]:.4f}, {l_rpy[1]:.4f}, {l_rpy[2]:.4f}),"
            )
            print(
                f"[A2OmniHandRetargeter][AUTO_CAL]     "
                f"right_fixed_rpy=({r_rpy[0]:.4f}, {r_rpy[1]:.4f}, {r_rpy[2]:.4f}),"
            )
            return

        if now - self._last_countdown_print >= 1.0:
            self._last_countdown_print = now
            print(
                f"[A2OmniHandRetargeter][AUTO_CAL] Stable {elapsed:.1f}/"
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
                f"[A2OmniHandRetargeter] wrist pre-compensation rpy (left, right)="
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
class A2OmniHandRetargeterCfg(RetargeterCfg):
    """Config for A2 + OmniHand hand retargeter."""

    enable_visualization: bool = False
    num_open_xr_hand_joints: int = 100
    hand_joint_names: list[str] | None = None
    left_fixed_rpy: tuple[float, float, float] = _DEFAULT_LEFT_RPY
    right_fixed_rpy: tuple[float, float, float] = _DEFAULT_RIGHT_RPY
    euler_convention: str = "xyz"
    debug_every: int = 0

    # Auto-calibration: when both wrists stay within `auto_calibrate_threshold_deg`
    # of their anchor for `auto_calibrate_seconds`, capture current rotations as
    # fixed_rot. 0 disables (manual cfg values used as-is).
    auto_calibrate_seconds: float = 0.0
    auto_calibrate_threshold_deg: float = 5.0

    left_shoulder_pos: tuple[float, float, float] = (-0.2, 0.0, 1.1)
    right_shoulder_pos: tuple[float, float, float] = (0.2, 0.0, 1.1)
    max_reach: float = 0.7
    pos_scale: float = 1.0

    retargeter_type: type[RetargeterBase] = A2OmniHandRetargeter
