# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard teleop adapter for A2 place-can-into-tray.

Wraps Se3Keyboard (6-DOF delta + gripper toggle) into the 38-dim action tensor
expected by pickplace_a2_env_cfg's bimanual Pink IK + 24 hand joints:

  action = [ left_wrist_pos(3), left_wrist_quat(4),
             right_wrist_pos(3), right_wrist_quat(4),
             left_hand_joints(12), right_hand_joints(12) ]

Left arm stays at a fixed idle pose. Right wrist accumulates keyboard deltas.
Right hand toggles between all-open and a preset grasp posture on the gripper
key (K by default).

Key bindings (inherited from Se3Keyboard):
  W/S: +/- x           A/D: +/- y           Q/E: +/- z
  Z/X: roll            T/G: pitch           C/V: yaw
  K:   toggle grasp    L:   reset accumulated pose
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from isaaclab.devices.device_base import DeviceBase, DeviceCfg
from isaaclab.devices.keyboard.se3_keyboard import Se3Keyboard, Se3KeyboardCfg


# Preset hand postures keyed by joint name so values always land on the right
# joints regardless of the order hand_joint_names is declared in. Closed angles
# are 95% of URDF upper limits for driven joints; mimic children derived via the
# URDF multipliers (thumb_2=0.4×thumb_1, thumb_3=0.6×thumb_1, finger_2=1.0×finger_1).
def _make_closed_posture(prefix: str) -> dict[str, float]:
    thumb_swing, thumb_1, finger_1 = 2.0, 0.7, 1.5
    return {
        f"{prefix}_thumb_swing_joint": thumb_swing,
        f"{prefix}_thumb_1_joint": thumb_1,
        f"{prefix}_thumb_2_joint": 0.40 * thumb_1,
        f"{prefix}_thumb_3_joint": 0.60 * thumb_1,
        f"{prefix}_index_1_joint": finger_1,
        f"{prefix}_index_2_joint": finger_1,
        f"{prefix}_middle_1_joint": finger_1,
        f"{prefix}_middle_2_joint": finger_1,
        f"{prefix}_ring_1_joint": finger_1,
        f"{prefix}_ring_2_joint": finger_1,
        f"{prefix}_pinky_1_joint": finger_1,
        f"{prefix}_pinky_2_joint": finger_1,
    }


_RIGHT_CLOSED = _make_closed_posture("R")
_LEFT_CLOSED = _make_closed_posture("L")


class A2KeyboardTeleop(DeviceBase):
    """Se3Keyboard-driven teleop that emits A2's 38-dim action tensor."""

    def __init__(self, cfg: A2KeyboardCfg):
        super().__init__()
        self._cfg = cfg
        self._sim_device = cfg.sim_device

        # The env's hand_joint_names order drives how the output hand-joint block is laid out;
        # values are looked up by name so this adapter works regardless of declaration order.
        if not cfg.hand_joint_names:
            raise ValueError("A2KeyboardCfg.hand_joint_names must be provided (pass env's _HAND_JOINTS).")
        self._hand_joint_names = list(cfg.hand_joint_names)

        # Internal Se3Keyboard handles raw key events (position/rotation deltas, gripper toggle).
        self._keyboard = Se3Keyboard(
            Se3KeyboardCfg(
                pos_sensitivity=cfg.pos_sensitivity,
                rot_sensitivity=cfg.rot_sensitivity,
                gripper_term=True,
                sim_device=cfg.sim_device,
            )
        )
        # Rebind L-key reset to our own (Se3Keyboard's reset doesn't touch our accumulators).
        self._keyboard.add_callback("L", self.reset)

        # Accumulated absolute right-wrist pose (world frame).
        self._right_pos = np.array(cfg.right_wrist_init_pos, dtype=np.float32)
        self._right_quat = np.array(cfg.right_wrist_init_quat, dtype=np.float32)  # [w, x, y, z]
        self._right_hand_closed = False
        self._last_gripper_sign = 1.0  # +1 open, -1 closed; detect edges

        # Fixed left-arm idle (out of the way, on the left side of the torso).
        self._left_pos = np.array(cfg.left_wrist_idle_pos, dtype=np.float32)
        self._left_quat = np.array(cfg.left_wrist_idle_quat, dtype=np.float32)

    # DeviceBase abstract methods ----------------------------------------------
    def reset(self) -> None:
        self._keyboard.reset()
        self._right_pos = np.array(self._cfg.right_wrist_init_pos, dtype=np.float32)
        self._right_quat = np.array(self._cfg.right_wrist_init_quat, dtype=np.float32)
        self._right_hand_closed = False
        self._last_gripper_sign = 1.0

    def add_callback(self, key: Any, func: Callable) -> None:
        # Forward arbitrary key bindings to the inner Se3Keyboard.
        self._keyboard.add_callback(key, func)

    def advance(self) -> torch.Tensor:
        raw = self._keyboard.advance().detach().cpu().numpy()  # 7-dim: [dx, dy, dz, rx, ry, rz, gripper]
        delta_pos = raw[0:3]
        delta_rot_vec = raw[3:6]
        gripper_sign = float(raw[6])

        # Accumulate position.
        self._right_pos = self._right_pos + delta_pos.astype(np.float32)

        # Accumulate rotation (delta_rot_vec is already scaled by rot_sensitivity).
        if np.linalg.norm(delta_rot_vec) > 1e-8:
            dR = Rotation.from_rotvec(delta_rot_vec)
            cur = Rotation.from_quat([self._right_quat[1], self._right_quat[2],
                                      self._right_quat[3], self._right_quat[0]])  # scipy xyzw
            new = dR * cur
            q_xyzw = new.as_quat()
            self._right_quat = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)

        # Detect gripper edge (Se3Keyboard toggles _close_gripper on each K press → sign flips).
        if gripper_sign != self._last_gripper_sign:
            self._right_hand_closed = not self._right_hand_closed
            self._last_gripper_sign = gripper_sign

        # Build hand-joint block by name lookup so this works for any
        # hand_joint_names ordering (must match asset order for PinkIK).
        hand_vals = np.zeros(len(self._hand_joint_names), dtype=np.float32)
        postures = {}
        # Left hand stays open (all zeros) — nothing to add.
        if self._right_hand_closed:
            postures.update(_RIGHT_CLOSED)
        for i, name in enumerate(self._hand_joint_names):
            if name in postures:
                hand_vals[i] = postures[name]

        action = np.concatenate([
            self._left_pos, self._left_quat,
            self._right_pos, self._right_quat,
            hand_vals,
        ]).astype(np.float32)

        return torch.tensor(action, dtype=torch.float32, device=self._sim_device)

    # Cleanup
    def __del__(self):
        try:
            del self._keyboard
        except Exception:
            pass


@dataclass
class A2KeyboardCfg(DeviceCfg):
    """Configuration for the A2 keyboard teleop adapter."""

    # Ordered list of hand joint names matching env_cfg._HAND_JOINTS (asset order).
    # Must match the same list the env's PinkInverseKinematicsAction uses so
    # action tensor slots line up with the joints Pink writes to.
    hand_joint_names: list[str] = field(default_factory=list)

    # Per-tick translation magnitude (meters). With default step_hz=30, 0.005 ≈ 15 cm/s.
    pos_sensitivity: float = 0.005
    # Per-tick rotation magnitude (radians). 0.02 ≈ 35°/s at 30 Hz.
    rot_sensitivity: float = 0.02

    # Initial right-wrist pose (reachable in front of the torso, just above table height).
    right_wrist_init_pos: tuple[float, float, float] = (-0.22, 0.30, 1.10)
    right_wrist_init_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # wxyz

    # Fixed left-wrist idle pose (mirrors right).
    left_wrist_idle_pos: tuple[float, float, float] = (0.22, 0.30, 1.10)
    left_wrist_idle_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    retargeters: list = field(default_factory=list)
    class_type: type[DeviceBase] = A2KeyboardTeleop
