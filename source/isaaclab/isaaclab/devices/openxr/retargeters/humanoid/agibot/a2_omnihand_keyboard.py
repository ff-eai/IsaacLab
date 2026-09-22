# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single-arm keyboard teleop for A2 + OmniHand.

Wraps Se3Keyboard so its 6-axis delta + gripper toggle drive the RIGHT arm's
absolute wrist pose plus a hand open/close state. Left arm stays at idle.
Output matches the env's action layout: left_wrist(7) + right_wrist(7) + hand(32).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from isaaclab.devices.device_base import DeviceBase
from isaaclab.devices.keyboard.se3_keyboard import Se3Keyboard, Se3KeyboardCfg


def _hand_open() -> np.ndarray:
    return np.zeros(32, dtype=np.float32)


def _hand_closed_right_only() -> np.ndarray:
    """Layout matches env's _HAND_JOINTS (16 L + 16 R)."""
    out = np.zeros(32, dtype=np.float32)
    mcp = 0.84
    pip = 1.48
    out[16] = 1.10           # R_thumb_roll
    out[17] = -1.50          # R_thumb_abad
    out[18] = mcp            # R_thumb_mcp
    out[19] = 0.0            # R_index_abad
    out[20] = pip            # R_index_pip
    out[21] = pip            # R_middle_pip
    out[22] = 0.0            # R_ring_abad
    out[23] = pip            # R_ring_pip
    out[24] = 0.0            # R_pinky_abad
    out[25] = pip            # R_pinky_pip
    out[26] = 1.33 * mcp     # R_thumb_pip (mimic)
    out[27] = 1.30 * mcp     # R_thumb_dip (mimic)
    out[28] = 1.097 * pip    # R_index_dip (mimic)
    out[29] = 1.097 * pip    # R_middle_dip (mimic)
    out[30] = 1.097 * pip    # R_ring_dip (mimic)
    out[31] = 1.097 * pip    # R_pinky_dip (mimic)
    return out


class A2OmniHandKeyboard(Se3Keyboard):
    """Se3Keyboard that emits the full 46-DoF A2+OmniHand action."""

    def __init__(self, cfg: "A2OmniHandKeyboardCfg"):
        super().__init__(cfg)
        self._cfg = cfg
        self._dt = cfg.dt
        self._left_wrist = np.asarray(cfg.left_wrist_idle, dtype=np.float32)
        self._right_pos = np.asarray(cfg.right_wrist_start[:3], dtype=np.float32)
        # Quat is (w, x, y, z) in env idle_action; scipy uses (x,y,z,w).
        qw, qx, qy, qz = cfg.right_wrist_start[3:]
        self._right_rot = R.from_quat([qx, qy, qz, qw])

    def reset(self):
        super().reset()
        self._right_pos = np.asarray(self._cfg.right_wrist_start[:3], dtype=np.float32)
        qw, qx, qy, qz = self._cfg.right_wrist_start[3:]
        self._right_rot = R.from_quat([qx, qy, qz, qw])

    def advance(self) -> torch.Tensor:
        # Parent emits [dx,dy,dz,rx,ry,rz,gripper] in *velocity* units; integrate.
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
class A2OmniHandKeyboardCfg(Se3KeyboardCfg):
    """Keyboard config for A2 + OmniHand single-arm teleop."""

    left_wrist_idle: tuple = (-0.22, 0.30, 1.10, 1.0, 0.0, 0.0, 0.0)
    right_wrist_start: tuple = (0.22, 0.30, 1.10, 1.0, 0.0, 0.0, 0.0)
    dt: float = 1.0 / 30.0
    class_type: type[DeviceBase] = A2OmniHandKeyboard
