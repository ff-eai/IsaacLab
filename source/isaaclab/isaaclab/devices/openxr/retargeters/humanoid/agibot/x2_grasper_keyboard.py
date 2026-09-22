# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard teleop for X2 + simple grasper.

Output matches the env action layout:
  - left wrist pose (7)
  - right wrist pose (7)
  - left grasper joints (2): open at 0, close at 0.7
  - right grasper joints (2): open at 0, close at 0.7
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from isaaclab.devices.device_base import DeviceBase
from isaaclab.devices.keyboard.se3_keyboard import Se3Keyboard, Se3KeyboardCfg


def _hand_open() -> np.ndarray:
    return np.zeros(4, dtype=np.float32)


def _hand_closed_right_only() -> np.ndarray:
    out = np.zeros(4, dtype=np.float32)
    out[2] = 0.7
    out[3] = 0.7
    return out


class X2GrasperKeyboard(Se3Keyboard):
    """Se3Keyboard that emits an 18-DoF action (two wrists + grasper joints)."""

    def __init__(self, cfg: "X2GrasperKeyboardCfg"):
        super().__init__(cfg)
        self._cfg = cfg
        self._dt = cfg.dt
        self._left_wrist = np.asarray(cfg.left_wrist_idle, dtype=np.float32)
        self._right_pos = np.asarray(cfg.right_wrist_start[:3], dtype=np.float32)
        qw, qx, qy, qz = cfg.right_wrist_start[3:]
        self._right_rot = R.from_quat([qx, qy, qz, qw])

    def reset(self):
        super().reset()
        self._right_pos = np.asarray(self._cfg.right_wrist_start[:3], dtype=np.float32)
        qw, qx, qy, qz = self._cfg.right_wrist_start[3:]
        self._right_rot = R.from_quat([qx, qy, qz, qw])

    def advance(self) -> torch.Tensor:
        cmd = super().advance().cpu().numpy()
        dpos = cmd[:3] * self._dt
        drot_vec = cmd[3:6] * self._dt
        gripper = cmd[6] if self._cfg.gripper_term else 1.0  # +1 open, -1 close

        self._right_pos = self._right_pos + dpos
        if np.any(drot_vec):
            self._right_rot = R.from_rotvec(drot_vec) * self._right_rot

        qx, qy, qz, qw = self._right_rot.as_quat()
        right_wrist = np.array(
            [self._right_pos[0], self._right_pos[1], self._right_pos[2], qw, qx, qy, qz],
            dtype=np.float32,
        )

        hand = _hand_open() if gripper >= 0 else _hand_closed_right_only()
        action = np.concatenate([self._left_wrist, right_wrist, hand]).astype(np.float32)
        return torch.tensor(action, dtype=torch.float32, device=self._sim_device)


@dataclass
class X2GrasperKeyboardCfg(Se3KeyboardCfg):
    """Keyboard config for X2 + grasper single-arm teleop."""

    left_wrist_idle: tuple = (-0.22, 0.30, 1.10, 1.0, 0.0, 0.0, 0.0)
    right_wrist_start: tuple = (0.22, 0.30, 1.10, 1.0, 0.0, 0.0, 0.0)
    dt: float = 1.0 / 30.0
    class_type: type[DeviceBase] = X2GrasperKeyboard

