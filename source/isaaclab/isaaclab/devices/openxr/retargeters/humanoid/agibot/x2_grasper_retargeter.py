# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OpenXR hand tracking → X2 + simple grasper retargeter.

Output layout matches the grasper env action vector:
    [left_wrist_pose (7), right_wrist_pose (7), grasper_joint_angles (4)]
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Final

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

import isaaclab.sim as sim_utils
from isaaclab.devices.device_base import DeviceBase
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from .a2_retargeter import _compose_openxr_wrist

_DEFAULT_LEFT_RPY = (0.0, 0.0, 0.0)
_DEFAULT_RIGHT_RPY = (0.0, 0.0, 3.1415927)


class X2GrasperRetargeter(RetargeterBase):
    """Retargets OpenXR hand tracking → X2 wrist poses + binary grasper joints."""

    GRIPPER_CLOSE_METERS: Final[float] = 0.03
    GRIPPER_OPEN_METERS: Final[float] = 0.05

    def __init__(self, cfg: "X2GrasperRetargeterCfg"):
        super().__init__(cfg)
        self._cfg = cfg
        self._grasper_joint_names = list(cfg.grasper_joint_names or [])
        self._enable_visualization = cfg.enable_visualization
        self._num_open_xr_hand_joints = cfg.num_open_xr_hand_joints
        self._sim_device = cfg.sim_device
        self._close_val = cfg.grasper_close_val

        self._left_fixed_rot = R.from_euler(cfg.euler_convention, cfg.left_fixed_rpy)
        self._right_fixed_rot = R.from_euler(cfg.euler_convention, cfg.right_fixed_rpy)
        self._logged_rpy = False
        self._debug_every = cfg.debug_every
        self._frame_count = 0

        self._auto_calibrate_seconds = float(cfg.auto_calibrate_seconds)
        self._auto_calibrate_threshold_rad = float(np.deg2rad(cfg.auto_calibrate_threshold_deg))
        self._calibrated = self._auto_calibrate_seconds <= 0.0
        self._stable_anchor_left: R | None = None
        self._stable_anchor_right: R | None = None
        self._stable_start_time: float | None = None
        self._last_countdown_print: float = 0.0

        self._left_gripper_closed = False
        self._right_gripper_closed = False

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
        lw, rw = np.asarray(left_wrist), np.asarray(right_wrist)
        lq = R.from_quat([lw[4], lw[5], lw[6], lw[3]])
        rq = R.from_quat([rw[4], rw[5], rw[6], rw[3]])

        now = time.monotonic()
        if self._stable_anchor_left is None:
            self._stable_anchor_left = lq
            self._stable_anchor_right = rq
            self._stable_start_time = now
            print(
                f"[X2GrasperRetargeter][AUTO_CAL] Watching for stillness — keep both palms "
                f"steady for {self._auto_calibrate_seconds:.1f}s..."
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
            print(f"[X2GrasperRetargeter][AUTO_CAL] CAPTURED after {elapsed:.1f}s stable.")
            print(
                f"[X2GrasperRetargeter][AUTO_CAL] left_fixed_rpy="
                f"({l_rpy[0]:.4f}, {l_rpy[1]:.4f}, {l_rpy[2]:.4f}),"
            )
            print(
                f"[X2GrasperRetargeter][AUTO_CAL] right_fixed_rpy="
                f"({r_rpy[0]:.4f}, {r_rpy[1]:.4f}, {r_rpy[2]:.4f}),"
            )
            return

        if now - self._last_countdown_print >= 1.0:
            self._last_countdown_print = now
            print(
                f"[X2GrasperRetargeter][AUTO_CAL] Stable {elapsed:.1f}/"
                f"{self._auto_calibrate_seconds:.1f}s "
                f"(drift L={np.degrees(rel_l):.1f}° R={np.degrees(rel_r):.1f}°)"
            )

    def _grasper_command(self, hand_poses: dict, previous_closed: bool) -> tuple[float, bool]:
        thumb_tip = hand_poses.get("thumb_tip")
        index_tip = hand_poses.get("index_tip")
        if thumb_tip is None or index_tip is None:
            return 0.0, previous_closed

        distance = float(np.linalg.norm(np.asarray(thumb_tip[:3]) - np.asarray(index_tip[:3])))
        closed = previous_closed
        if distance > self.GRIPPER_OPEN_METERS:
            closed = False
        elif distance < self.GRIPPER_CLOSE_METERS:
            closed = True
        return (self._close_val if closed else 0.0), closed

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

        left_val, self._left_gripper_closed = self._grasper_command(left_hand_poses, self._left_gripper_closed)
        right_val, self._right_gripper_closed = self._grasper_command(right_hand_poses, self._right_gripper_closed)

        grasper_joints = np.zeros(len(self._grasper_joint_names), dtype=np.float32)
        for idx, name in enumerate(self._grasper_joint_names):
            if name.startswith("L_"):
                grasper_joints[idx] = left_val
            elif name.startswith("R_"):
                grasper_joints[idx] = right_val

        if not self._logged_rpy:
            print(
                f"[X2GrasperRetargeter] wrist pre-compensation rpy (left, right)="
                f"{self._left_fixed_rot.as_euler('xyz').tolist()}, "
                f"{self._right_fixed_rot.as_euler('xyz').tolist()}"
            )
            self._logged_rpy = True

        left_target = _compose_openxr_wrist(
            np.asarray(left_wrist),
            self._left_fixed_rot,
            shoulder_pos=self._cfg.left_shoulder_pos,
            max_reach=self._cfg.max_reach,
            pos_scale=self._cfg.pos_scale,
        )
        right_target = _compose_openxr_wrist(
            np.asarray(right_wrist),
            self._right_fixed_rot,
            shoulder_pos=self._cfg.right_shoulder_pos,
            max_reach=self._cfg.max_reach,
            pos_scale=self._cfg.pos_scale,
        )

        self._frame_count += 1

        return torch.cat(
            [
                torch.tensor(left_target, dtype=torch.float32, device=self._sim_device),
                torch.tensor(right_target, dtype=torch.float32, device=self._sim_device),
                torch.tensor(grasper_joints, dtype=torch.float32, device=self._sim_device),
            ]
        )

    def get_requirements(self) -> list[RetargeterBase.Requirement]:
        return [RetargeterBase.Requirement.HAND_TRACKING]


@dataclass
class X2GrasperRetargeterCfg(RetargeterCfg):
    """Config for X2 + grasper hand retargeter."""

    enable_visualization: bool = False
    num_open_xr_hand_joints: int = 52
    grasper_joint_names: list[str] | None = None
    grasper_close_val: float = 0.7
    left_fixed_rpy: tuple[float, float, float] = _DEFAULT_LEFT_RPY
    right_fixed_rpy: tuple[float, float, float] = _DEFAULT_RIGHT_RPY
    euler_convention: str = "xyz"
    debug_every: int = 0
    auto_calibrate_seconds: float = 0.0
    auto_calibrate_threshold_deg: float = 5.0
    left_shoulder_pos: tuple[float, float, float] = (-0.2, 0.05, 1.07)
    right_shoulder_pos: tuple[float, float, float] = (0.2, 0.05, 1.07)
    max_reach: float = 0.7
    pos_scale: float = 1.0
    retargeter_type: type[RetargeterBase] = X2GrasperRetargeter
