# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A2 dex retargeting utility — ports gr1_t2_dex_retargeting_utils for Agibot A2 s6_hand.

Maps OpenXR hand tracking (26 joints per hand) → DexPilot IK over A2 s6_hand URDF →
6 driven joint angles per hand (thumb_swing, thumb_1, index_1, middle_1, ring_1, pinky_1).

The returned joint names are the DRIVEN joints only. Mimic-child joints (thumb_2,
thumb_3, each *_2_joint) are set downstream by the controller from the URDF mimic
multipliers.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import torch
from dex_retargeting.retargeting_config import RetargetingConfig
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)

# OpenXR hand has 26 joints; this index set maps to the 21-joint MANO-style input
# expected by DexPilot. Same indexing as the GR1T2 reference.
_HAND_JOINTS_INDEX = [1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25]

# Legacy empirical matrix (no longer used — replaced by auto-calibration at
# first frame in convert_hand_joints). Kept for fallback if auto-calibration
# fails: (x, y, z) → (-z, x, y).
_OPERATOR2MANO_RIGHT = np.array([[0, 1, 0], [0, 0, 1], [-1, 0, 0]])
_OPERATOR2MANO_LEFT = np.array([[0, 1, 0], [0, 0, 1], [-1, 0, 0]])


def _calibrate_operator2mano(joints_wrist_local: np.ndarray, handedness: str) -> np.ndarray:
    """Derive operator2mano so user's MCPs line up with A2's URDF wrist-frame axes.

    A2 URDF wrist-frame convention (verified via pinocchio FK on L/R_hand_nomimic.urdf):
      +X  spread direction (index side → pinky side; thumb at -X)
      +Y  palm depth (toward the back of the hand)
      +Z  fingers forward (wrist → middle MCP)

    This intentionally differs from MANO canonical — the dex-retargeting solver
    uses URDF FK for robot tip positions, so ref_values from the user's hand must
    live in the same frame as the URDF, not in an idealized MANO frame.

    handedness: "L" or "R" — controls the palm-normal (+Y) sign so it points out
    of the BACK of the hand for both hands.
    """
    middle_mcp = joints_wrist_local[9]
    index_mcp = joints_wrist_local[5]
    pinky_mcp = joints_wrist_local[17]

    # +Z_urdf = fingers forward
    z_axis = middle_mcp.astype(np.float64)
    z_axis /= np.linalg.norm(z_axis) + 1e-8

    # +X_urdf = spread (index → pinky) — this gives thumb side at -X for both hands.
    spread = (pinky_mcp - index_mcp).astype(np.float64)
    # Project spread onto the plane perpendicular to z_axis so it's clean.
    spread = spread - np.dot(spread, z_axis) * z_axis
    x_axis = spread / (np.linalg.norm(spread) + 1e-8)

    # +Y_urdf = cross(Z, X) = palm-back direction (right-hand rule)
    y_axis = np.cross(z_axis, x_axis)
    if handedness == "L":
        # For the left hand, this cross points INTO the palm; flip so +Y is the back.
        y_axis = -y_axis
        # Recompute x_axis to keep the frame right-handed after flipping y.
        x_axis = np.cross(y_axis, z_axis)

    R_axes = np.stack([x_axis, y_axis, z_axis], axis=1)  # columns are URDF axes in wrist-local
    # Row-vector convention (v @ M): v_urdf = v_local @ R_axes
    return R_axes.astype(np.float32)


_LEFT_DRIVEN = [
    "L_thumb_swing_joint",
    "L_thumb_1_joint",
    "L_index_1_joint",
    "L_middle_1_joint",
    "L_ring_1_joint",
    "L_pinky_1_joint",
    "L_index_2_joint",
    "L_middle_2_joint",
    "L_ring_2_joint",
    "L_pinky_2_joint",
]


_RIGHT_DRIVEN = [
    "R_thumb_swing_joint",
    "R_thumb_1_joint",
    "R_index_1_joint",
    "R_middle_1_joint",
    "R_ring_1_joint",
    "R_pinky_1_joint",
    "R_index_2_joint",
    "R_middle_2_joint",
    "R_ring_2_joint",
    "R_pinky_2_joint",
]


# Geometric retargeting: bypass dex_retargeting's solver (which gets stuck in
# per-finger local minima during live streaming) and compute each finger's
# closure directly from OpenXR geometry. Deterministic, can't get stuck.
#
# For each finger, compute the ratio of (tip-to-wrist distance) to the typical
# extended length. When extended, ratio ≈ 1. When curled, ratio drops. Map
# inversely to joint_1 in its range [0, 1.7] rad (thumb_1 has range [0, 0.78]).
#
# OpenXR 21-joint MANO-style layout (after _HAND_JOINTS_INDEX picking):
#   0:     wrist
#   4:     thumb_tip
#   5:     index_proximal  (first knuckle)
#   8:     index_tip
#   9:     middle_proximal
#   12:    middle_tip
#   13:    ring_proximal
#   16:    ring_tip
#   17:    pinky_proximal
#   20:    pinky_tip
#   (thumb metacarpal is at 1)
_FINGER_PROXIMAL_IDX = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}
# MANO layout per finger: proximal (MCP, at PROXIMAL_IDX), intermediate (PIP, +1),
# distal (DIP, +2), tip (+3). For per-joint retargeting we need all four.


_geo_debug_counter = {"count": 0}


def _angle_between(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    """Unsigned angle (radians) between two 3-vectors."""
    an = a / (np.linalg.norm(a) + eps)
    bn = b / (np.linalg.norm(b) + eps)
    return float(np.arccos(np.clip(np.dot(an, bn), -1.0, 1.0)))


def _geometric_hand_closure(joint_position_mano: np.ndarray, side: str) -> np.ndarray:
    """Return 10 driven-joint values for one hand in the order:
      [thumb_swing, thumb_1,
       index_1, middle_1, ring_1, pinky_1,
       index_2, middle_2, ring_2, pinky_2].

    Per-joint retargeting: MCP flexion drives joint_1, PIP flexion drives joint_2,
    computed from MANO joint positions — so a hook (only PIP curled) and a fist
    (MCP+PIP both curled) look different on the robot.

    joint_position_mano: shape (21, 3) — OpenXR joints after _HAND_JOINTS_INDEX
    picking, expressed in wrist frame.
    """
    out = np.zeros(10, dtype=np.float32)

    wrist = joint_position_mano[0]
    # Palm-forward reference: wrist → middle finger MCP (joint 9). This points
    # along the hand's long axis regardless of hand orientation.
    palm_fwd = joint_position_mano[9] - wrist

    finger_joint_limit = 1.7  # joint_1 and joint_2 URDF upper limit
    # Empirical flexion range [extended_angle, curled_angle] in radians,
    # normalized to [0, 1]. Slight margin at the low end so a naturally-relaxed
    # finger (not truly at 0° to palm_fwd) doesn't register as curled.
    mcp_ext, mcp_curl = 0.35, 1.5   # MCP flexion range
    pip_ext, pip_curl = 0.25, 1.5   # PIP flexion range (tighter lower bound)

    for i, finger in enumerate(["index", "middle", "ring", "pinky"]):
        p = _FINGER_PROXIMAL_IDX[finger]
        mcp_pos = joint_position_mano[p]
        pip_pos = joint_position_mano[p + 1]
        dip_pos = joint_position_mano[p + 2]
        seg1 = pip_pos - mcp_pos
        seg2 = dip_pos - pip_pos

        mcp_angle = _angle_between(palm_fwd, seg1)
        pip_angle = _angle_between(seg1, seg2)

        mcp_t = np.clip((mcp_angle - mcp_ext) / (mcp_curl - mcp_ext), 0.0, 1.0)
        pip_t = np.clip((pip_angle - pip_ext) / (pip_curl - pip_ext), 0.0, 1.0)

        out[2 + i] = mcp_t * finger_joint_limit        # index_1, middle_1, ring_1, pinky_1
        out[6 + i] = pip_t * finger_joint_limit        # index_2, middle_2, ring_2, pinky_2

    # Thumb: two DOF to compute.
    # thumb_1 joint (curl): tip-to-metacarpal distance metric.
    thumb_metacarpal = joint_position_mano[1]
    thumb_tip = joint_position_mano[4]
    thumb_curl_dist = np.linalg.norm(thumb_tip - thumb_metacarpal)
    thumb_ext, thumb_curl_min = 0.090, 0.082
    thumb_t_linear = np.clip((thumb_ext - thumb_curl_dist) / (thumb_ext - thumb_curl_min), 0.0, 1.0)
    # Pico's thumb tracking compresses at flexion — rest pose already sits
    # near the top of the linear signal. A power curve holds the output low
    # through the rest band and ramps sharply toward full closure only on a
    # real fist. Higher gamma = more "dead zone" at rest.
    thumb_gamma = 5.0
    thumb_t = thumb_t_linear ** thumb_gamma
    out[1] = thumb_t * 0.78  # thumb_1 joint limit [0, 0.78]

    # thumb_swing (thumb opposition): angle between the thumb's proximal segment
    # (CMC → MCP, joints 1 → 2) and the palm-forward direction. The CMC joint
    # rotates the *thumb's long axis* relative to the hand — measuring the joint
    # base position (wrist→CMC) didn't capture that rotation, which is why
    # opposition wasn't tracking. Using the segment vector does.
    thumb_mcp = joint_position_mano[2]
    v_thumb_seg = thumb_mcp - thumb_metacarpal
    angle = _angle_between(v_thumb_seg, palm_fwd)
    # Empirical range: splayed thumb ≈ 0.5 rad to palm_fwd; fully opposed ≈ 1.4 rad.
    swing_t = np.clip((angle - 0.5) / (1.4 - 0.5), 0.0, 1.0)
    out[0] = swing_t * 2.286

    _geo_debug_counter["count"] += 1
    if _geo_debug_counter["count"] % 60 == 0:
        print(
            f"[A2Geo] {side} thumb_curl={thumb_curl_dist:.3f}m "
            f"thumb_1={out[1]:.2f} thumb_swing={out[0]:.2f}  "
            f"idx1={out[2]:.2f}/2={out[6]:.2f} mid1={out[3]:.2f}/2={out[7]:.2f} "
            f"ring1={out[4]:.2f}/2={out[8]:.2f} pky1={out[5]:.2f}/2={out[9]:.2f}"
        )

    return out

# Mimic coupling from A2.urdf (child = driver × multiplier). Applied in the
# retargeter so the nomimic USD (where mimic joints are expanded to independent
# joints) still receives consistent targets for child joints each step.
# Mimic dicts. _GEO is used when USE_GEOMETRIC=True (per-joint finger tracking
# drives index_2/.../pinky_2 directly, so they're not mimicked). _DEX is used
# when USE_GEOMETRIC=False (dex solver only returns joint_1 values for fingers;
# joint_2 must follow via 1.0x mimic like the URDF declares).
_LEFT_MIMIC_GEO = {
    "L_thumb_2_joint": ("L_thumb_1_joint", 0.40),
    "L_thumb_3_joint": ("L_thumb_1_joint", 0.60),
}
# Enforce URDF mimic ratios after dex output: the nomimic URDF lets dex solve
# thumb_2/thumb_3 freely, and L/R asymmetric joint limits made R thumb_3
# saturate to ~π/2 while L stayed near 0. Override to URDF mimic ratios.
_LEFT_MIMIC_DEX: dict[str, tuple[str, float]] = {
    "L_thumb_2_joint": ("L_thumb_1_joint", 0.40),
    "L_thumb_3_joint": ("L_thumb_1_joint", 0.60),
}
_RIGHT_MIMIC_GEO = {
    "R_thumb_2_joint": ("R_thumb_1_joint", 0.40),
    "R_thumb_3_joint": ("R_thumb_1_joint", 0.60),
}
_RIGHT_MIMIC_DEX: dict[str, tuple[str, float]] = {
    "R_thumb_2_joint": ("R_thumb_1_joint", 0.40),
    "R_thumb_3_joint": ("R_thumb_1_joint", 0.60),
}


def _expand_with_mimic(driven_names: list[str], driven_vals: np.ndarray, mimic: dict[str, tuple[str, float]]):
    """Return (full_names, full_vals) where full = driven + mimic children with vals populated."""
    name_to_val = dict(zip(driven_names, driven_vals))
    full_names = list(driven_names)
    full_vals = list(driven_vals)
    for child, (driver, mult) in mimic.items():
        full_names.append(child)
        full_vals.append(name_to_val[driver] * mult)
    return full_names, np.array(full_vals, dtype=np.float32)


class A2DexRetargeting:
    """OpenXR → A2 s6_hand retargeting (driven joints only)."""

    def __init__(
        self,
        hand_joint_names: list[str],
        left_hand_config_filename: str = "a2_hand_left_dexpilot.yml",
        right_hand_config_filename: str = "a2_hand_right_dexpilot.yml",
    ) -> None:
        """Args:
            hand_joint_names: Full set of Isaac Lab hand joint names (order defines output indexing).
            left_hand_config_filename, right_hand_config_filename: YAML configs under data/configs/dex-retargeting/.
        """
        config_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "data", "configs", "dex-retargeting")
        )
        left_config_path = os.path.join(config_dir, left_hand_config_filename)
        right_config_path = os.path.join(config_dir, right_hand_config_filename)

        self._dex_left_hand = RetargetingConfig.load_from_file(left_config_path).build()
        self._dex_right_hand = RetargetingConfig.load_from_file(right_config_path).build()

        # Driven joint names from dex_retargeting. For A2 these match _LEFT_DRIVEN / _RIGHT_DRIVEN.
        self._left_driven_names = self._dex_left_hand.optimizer.robot.dof_joint_names
        self._right_driven_names = self._dex_right_hand.optimizer.robot.dof_joint_names
        # Full output names (driven + mimic children) — retargeter emits values for all of these.
        # Initial layout assumes DEX path (no per-finger joint_2 in driven); the
        # GEO path overrides these in compute_left/compute_right.
        self.left_dof_names = list(self._left_driven_names) + list(_LEFT_MIMIC_DEX.keys())
        self.right_dof_names = list(self._right_driven_names) + list(_RIGHT_MIMIC_DEX.keys())
        self.dof_names = self.left_dof_names + self.right_dof_names
        self.isaac_lab_hand_joint_names = hand_joint_names

        logger.info("[A2DexRetargeter] init done. left_out=%d right_out=%d",
                    len(self.left_dof_names), len(self.right_dof_names))

    def convert_hand_joints(
        self, hand_poses: dict[str, np.ndarray], operator2mano: np.ndarray, side: str = "R"
    ) -> np.ndarray:
        """OpenXR 26 joints → MANO-style 21 joints in canonical wrist frame.

        First call per side auto-calibrates operator2mano from the actual MCP
        geometry; subsequent calls reuse the cached matrix. Pass `side="L"` or
        `side="R"` so the calibrator picks the right palm-normal sign.
        """
        joint_position = np.zeros((21, 3))
        hand_joints = list(hand_poses.values())
        for i, idx in enumerate(_HAND_JOINTS_INDEX):
            joint_position[i] = hand_joints[idx][:3]
        # Relative to wrist (origin).
        joint_position = joint_position - joint_position[0:1, :]
        xr_wrist_quat = hand_poses.get("wrist")[3:]
        # OpenXR quat is (w, x, y, z); scipy expects (x, y, z, w).
        wrist_rot = R.from_quat(
            [xr_wrist_quat[1], xr_wrist_quat[2], xr_wrist_quat[3], xr_wrist_quat[0]]
        ).as_matrix()
        wrist_local = joint_position @ wrist_rot

        cache_attr = f"_op2mano_cached_{side}"
        cached = getattr(self, cache_attr, None)
        if cached is None:
            # Only calibrate once we have plausible finger positions (not all zeros).
            middle_mcp_mag = float(np.linalg.norm(wrist_local[9]))
            if middle_mcp_mag > 0.02:  # at least 2cm — real hand data
                cached = _calibrate_operator2mano(wrist_local, side)
                setattr(self, cache_attr, cached)
            else:
                # No real data yet — fall back to the legacy static matrix for this frame.
                cached = operator2mano
        return wrist_local @ cached

    def compute_ref_value(
        self, joint_position: np.ndarray, indices: np.ndarray, retargeting_type: str
    ) -> np.ndarray:
        if retargeting_type == "POSITION":
            return joint_position[indices, :]
        origin_indices = indices[0, :]
        task_indices = indices[1, :]
        return joint_position[task_indices, :] - joint_position[origin_indices, :]

    def compute_one_hand(
        self,
        hand_joints: dict[str, np.ndarray],
        retargeting: RetargetingConfig,
        operator2mano: np.ndarray,
    ) -> np.ndarray:
        side = "L" if operator2mano is _OPERATOR2MANO_LEFT else "R"
        joint_pos = self.convert_hand_joints(hand_joints, operator2mano, side)
        ref_value = self.compute_ref_value(
            joint_pos,
            indices=retargeting.optimizer.target_link_human_indices,
            retargeting_type=retargeting.optimizer.retargeting_type,
        )
        with torch.enable_grad():
            with torch.inference_mode(False):
                return retargeting.retarget(ref_value)

    def get_joint_names(self) -> list[str]:
        return self.dof_names

    def get_left_joint_names(self) -> list[str]:
        return self.left_dof_names

    def get_right_joint_names(self) -> list[str]:
        return self.right_dof_names

    # GR1T2 finger rule: pure DexPilot, each hand uses its own native MANO
    # transform (compute_left → _OPERATOR2MANO_LEFT, compute_right →
    # _OPERATOR2MANO_RIGHT). Geometric path is preserved below for fallback
    # but disabled by default to match GR1T2's working pipeline.
    USE_GEOMETRIC = False

    def _geometric_compute(self, hand_poses, operator2mano, driven_names, mimic, side):
        """Simple, deterministic finger closure from OpenXR geometry — no solver."""
        joint_pos = self.convert_hand_joints(hand_poses, operator2mano, side)
        driven_vals = _geometric_hand_closure(joint_pos, side)
        # driven_names order = _LEFT_DRIVEN or _RIGHT_DRIVEN
        # driven_vals order = [thumb_swing, thumb_1, index_1, middle_1, ring_1, pinky_1]
        # These happen to match, but be explicit:
        expected = _LEFT_DRIVEN if side == "L" else _RIGHT_DRIVEN
        assert list(driven_names) == expected, f"driven_names mismatch: {driven_names} vs {expected}"
        _, full_vals = _expand_with_mimic(list(driven_names), driven_vals, mimic)
        return full_vals

    def compute_left(self, left_hand_poses: dict[str, np.ndarray] | None) -> np.ndarray:
        """Returns values for driven + mimic-child joints (left), aligned with self.left_dof_names."""
        if left_hand_poses is None:
            return np.zeros(len(self.left_dof_names), dtype=np.float32)
        if self.USE_GEOMETRIC:
            # Override dex's left_dof_names with our fixed _LEFT_DRIVEN ordering.
            self.left_dof_names = list(_LEFT_DRIVEN) + list(_LEFT_MIMIC_GEO.keys())
            return self._geometric_compute(
                left_hand_poses, _OPERATOR2MANO_LEFT, _LEFT_DRIVEN, _LEFT_MIMIC_GEO, "L",
            )
        self.left_dof_names = list(self._left_driven_names) + list(_LEFT_MIMIC_DEX.keys())
        driven = self.compute_one_hand(left_hand_poses, self._dex_left_hand, _OPERATOR2MANO_LEFT)
        _, full_vals = _expand_with_mimic(list(self._left_driven_names), np.asarray(driven), _LEFT_MIMIC_DEX)
        return full_vals

    def compute_right(self, right_hand_poses: dict[str, np.ndarray] | None) -> np.ndarray:
        """Returns values for driven + mimic-child joints (right), aligned with self.right_dof_names."""
        if right_hand_poses is None:
            return np.zeros(len(self.right_dof_names), dtype=np.float32)
        if self.USE_GEOMETRIC:
            self.right_dof_names = list(_RIGHT_DRIVEN) + list(_RIGHT_MIMIC_GEO.keys())
            return self._geometric_compute(
                right_hand_poses, _OPERATOR2MANO_RIGHT, _RIGHT_DRIVEN, _RIGHT_MIMIC_GEO, "R",
            )
        self.right_dof_names = list(self._right_driven_names) + list(_RIGHT_MIMIC_DEX.keys())
        driven = self.compute_one_hand(right_hand_poses, self._dex_right_hand, _OPERATOR2MANO_RIGHT)
        _, full_vals = _expand_with_mimic(list(self._right_driven_names), np.asarray(driven), _RIGHT_MIMIC_DEX)
        return full_vals
