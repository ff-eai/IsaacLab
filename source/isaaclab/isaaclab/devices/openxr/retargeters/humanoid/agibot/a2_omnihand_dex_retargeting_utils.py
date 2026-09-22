# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OmniHand T2 dex retargeter for A2 humanoid — mirrors GR1T2's minimal pipeline.

Drives 10 DoF per hand via dex-retargeting's DexPilot solver. Uses a fixed
canonical-MANO operator2mano matrix (same as GR1T2) and pure upstream defaults
for smoothing/scale. No custom calibration, per-frame reset, or mimic expansion
— the dex config's URDF handles mimic internally.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import torch
from dex_retargeting.retargeting_config import RetargetingConfig
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)


# OpenXR 26 → MANO-style 21 index pick.
_HAND_JOINTS_INDEX = [1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25]

# Canonical MANO operator2mano, rotated from the GR1T2 default so that the
# user's middle_MCP lands along +X (canonical "fingers forward"). Verified
# from live telemetry: with the old matrix, middle_MCP was at (0, 0, +0.084)
# — fingers forward on +Z instead of +X — which broke the DexPilot solver
# (targets in wrong frame → solver converges to all-curled).
_OPERATOR2MANO_RIGHT = np.array(
    [
        [0, 1, 0],
        [0, 0, -1],
        [-1, 0, 0],
    ]
)
# Left = original (det=+1, rotation-not-mirror). True reflections (det=-1)
# break the joint chain geometry — kinks collapse to 90°. The geometric
# retargeter compensates for the L chirality below in `_omnihand_geometric`
# via a side-aware splay offset.
_OPERATOR2MANO_LEFT = np.array(
    [
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
    ]
)


_LEFT_HAND_JOINT_NAMES = [
    "L_thumb_roll_joint",
    "L_thumb_abad_joint",
    "L_thumb_mcp_joint",
    "L_index_abad_joint",
    "L_index_pip_joint",
    "L_middle_pip_joint",
    "L_ring_abad_joint",
    "L_ring_pip_joint",
    "L_pinky_abad_joint",
    "L_pinky_pip_joint",
]

_RIGHT_HAND_JOINT_NAMES = [
    "R_thumb_roll_joint",
    "R_thumb_abad_joint",
    "R_thumb_mcp_joint",
    "R_index_abad_joint",
    "R_index_pip_joint",
    "R_middle_pip_joint",
    "R_ring_abad_joint",
    "R_ring_pip_joint",
    "R_pinky_abad_joint",
    "R_pinky_pip_joint",
]


class A2OmniHandDexRetargeting:
    """OpenXR → A2+OmniHand retargeter (DexPilot, GR1T2-style minimal pipeline)."""

    def __init__(
        self,
        hand_joint_names: list[str],
        left_config: str = "a2_omnihand_left_dexpilot.yml",
        right_config: str = "a2_omnihand_right_dexpilot.yml",
    ):
        config_dir = os.path.join(
            os.path.dirname(__file__), "data", "configs", "dex-retargeting"
        )
        left_path = os.path.join(config_dir, left_config)
        right_path = os.path.join(config_dir, right_config)
        self._dex_left_hand = RetargetingConfig.load_from_file(left_path).build()
        self._dex_right_hand = RetargetingConfig.load_from_file(right_path).build()

        self.left_dof_names = self._dex_left_hand.optimizer.robot.dof_joint_names
        self.right_dof_names = self._dex_right_hand.optimizer.robot.dof_joint_names
        self.dof_names = self.left_dof_names + self.right_dof_names
        self.isaac_lab_hand_joint_names = hand_joint_names

        logger.info("[A2OmniHandDex] init done.")

    def convert_hand_joints(self, hand_poses: dict[str, np.ndarray], operator2mano: np.ndarray) -> np.ndarray:
        """OpenXR 26 joints → 21 MANO-style joints in canonical wrist frame."""
        joint_position = np.zeros((21, 3))
        hand_joints = list(hand_poses.values())
        for i in range(len(_HAND_JOINTS_INDEX)):
            joint_position[i] = hand_joints[_HAND_JOINTS_INDEX[i]][:3]

        # Convert hand pose to the canonical frame.
        joint_position = joint_position - joint_position[0:1, :]
        xr_wrist_quat = hand_poses.get("wrist")[3:]
        # OpenXR uses w,x,y,z quat; scipy takes x,y,z,w.
        wrist_rot = R.from_quat(
            [xr_wrist_quat[1], xr_wrist_quat[2], xr_wrist_quat[3], xr_wrist_quat[0]]
        ).as_matrix()
        return joint_position @ wrist_rot @ operator2mano

    def compute_ref_value(self, joint_position: np.ndarray, indices: np.ndarray, retargeting_type: str) -> np.ndarray:
        if retargeting_type == "POSITION":
            return joint_position[indices, :]
        origin_indices = indices[0, :]
        task_indices = indices[1, :]
        return joint_position[task_indices, :] - joint_position[origin_indices, :]

    _debug_counter = 0

    def compute_one_hand(
        self, hand_joints: dict[str, np.ndarray], retargeting, operator2mano: np.ndarray
    ) -> np.ndarray:
        joint_pos = self.convert_hand_joints(hand_joints, operator2mano)
        ref_value = self.compute_ref_value(
            joint_pos,
            indices=retargeting.optimizer.target_link_human_indices,
            retargeting_type=retargeting.optimizer.retargeting_type,
        )
        with torch.enable_grad():
            with torch.inference_mode(False):
                driven = retargeting.retarget(ref_value)
        return driven

    def get_joint_names(self) -> list[str]:
        return self.dof_names

    def get_left_joint_names(self) -> list[str]:
        return self.left_dof_names

    def get_right_joint_names(self) -> list[str]:
        return self.right_dof_names

    # GR1T2 finger rule: pure DexPilot, each hand uses its own native MANO
    # transform. Geometric fallback retained below for diagnostic use but
    # disabled by default to match GR1T2's working pipeline.
    USE_GEOMETRIC = False

    def compute_left(self, left_hand_poses: dict[str, np.ndarray] | None) -> np.ndarray:
        if left_hand_poses is None:
            return np.zeros(len(_LEFT_HAND_JOINT_NAMES))
        if self.USE_GEOMETRIC:
            # Geometric formulas in `_omnihand_geometric` were tuned for the R
            # hand's chirality (thumb on +Y side of palm_fwd). Two-step fix for
            # the L hand:
            #   1. Apply R's MANO transform (so the joint chain layout matches R).
            #   2. Negate Y to mirror the L hand into R-hand chirality. This
            #      moves the L thumb onto the same palm-side as the R thumb,
            #      making `thumb_total` and `thumb_splay` behave identically to
            #      the R hand for an equivalent physical pose.
            joint_pos = self.convert_hand_joints(left_hand_poses, _OPERATOR2MANO_RIGHT).copy()
            joint_pos[:, 1] = -joint_pos[:, 1]
            return _omnihand_geometric(joint_pos, "L")
        return self.compute_one_hand(left_hand_poses, self._dex_left_hand, _OPERATOR2MANO_LEFT)

    def compute_right(self, right_hand_poses: dict[str, np.ndarray] | None) -> np.ndarray:
        if right_hand_poses is None:
            return np.zeros(len(_RIGHT_HAND_JOINT_NAMES))
        if self.USE_GEOMETRIC:
            joint_pos = self.convert_hand_joints(right_hand_poses, _OPERATOR2MANO_RIGHT)
            return _omnihand_geometric(joint_pos, "R")
        return self.compute_one_hand(right_hand_poses, self._dex_right_hand, _OPERATOR2MANO_RIGHT)


def _angle_between(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    an = a / (np.linalg.norm(a) + eps)
    bn = b / (np.linalg.norm(b) + eps)
    return float(np.arccos(np.clip(np.dot(an, bn), -1.0, 1.0)))


def _omnihand_geometric(joint_mano: np.ndarray, side: str) -> np.ndarray:
    """Direct geometric mapping MANO joints → OmniHand 10 driven DoF.

    Output order matches _{LEFT,RIGHT}_HAND_JOINT_NAMES:
      [thumb_roll, thumb_abad, thumb_mcp,
       index_abad, index_pip,
       middle_pip,
       ring_abad,  ring_pip,
       pinky_abad, pinky_pip]

    Assumes MANO is in canonical frame (fingers along +X) after operator2mano.
    """
    out = np.zeros(10, dtype=np.float32)

    wrist = joint_mano[0]
    middle_mcp = joint_mano[9]
    palm_fwd = middle_mcp - wrist
    palm_fwd /= np.linalg.norm(palm_fwd) + 1e-8

    # --- Thumb ---
    thumb_cmc = joint_mano[1]
    thumb_mcp_pos = joint_mano[2]
    thumb_ip_pos = joint_mano[3]
    thumb_tip_pos = joint_mano[4]

    # Thumb total curl = MCP-kink + IP-kink. Single-joint kinks from XR are
    # noisy and small (telemetry shows ~0.05 rad even for full fist), so
    # combine both and use total flex as driver.
    v_cmc_mcp = thumb_mcp_pos - thumb_cmc
    v_mcp_ip = thumb_ip_pos - thumb_mcp_pos
    v_ip_tip = thumb_tip_pos - thumb_ip_pos
    thumb_mcp_kink = _angle_between(v_cmc_mcp, v_mcp_ip)
    thumb_ip_kink = _angle_between(v_mcp_ip, v_ip_tip)
    thumb_total = thumb_mcp_kink + thumb_ip_kink
    # Live telemetry: extended thumb total ≈ 0.13 rad, full curl ≈ 0.85 rad.
    # Map to URDF mcp limit 0.84 (pip+dip mimic). Same map for both sides —
    # `compute_left` mirrors the L hand into R chirality before this runs.
    out[2] = np.clip((thumb_total - 0.1) / 0.6, 0.0, 1.0) * 0.84

    # thumb_abad / thumb_roll: opposition driven by where the thumb sits
    # relative to the palm. Use the angle between thumb base direction
    # (CMC→MCP) and palm_fwd as a robust proxy.
    thumb_dir = v_cmc_mcp / (np.linalg.norm(v_cmc_mcp) + 1e-8)
    thumb_splay = _angle_between(thumb_dir, palm_fwd)
    # Live telemetry: alongside ≈ 0.55 rad, opposed ≈ 0.95 rad.
    # URDF abad [-1.64, 0.045]; negative = opposition. URDF roll limit ≈ 1.12.
    splay_norm = np.clip((thumb_splay - 0.4) / 0.5, 0.0, 1.0)
    out[1] = -splay_norm * 1.5
    out[0] = splay_norm * 1.1

    # --- Fingers ---
    # MANO indices per finger: [MCP, PIP, DIP, TIP]
    finger_table = [
        ("index",  [5, 6, 7, 8],     3, 4),
        ("middle", [9, 10, 11, 12],  None, 5),
        ("ring",   [13, 14, 15, 16], 6, 7),
        ("pinky",  [17, 18, 19, 20], 8, 9),
    ]
    for name, idx, abad_idx, pip_idx in finger_table:
        mcp_pos = joint_mano[idx[0]]
        pip_pos_j = joint_mano[idx[1]]
        dip_pos_j = joint_mano[idx[2]]
        v_mp = pip_pos_j - mcp_pos      # proximal phalanx (MCP→PIP)
        v_pd = dip_pos_j - pip_pos_j    # middle phalanx (PIP→DIP)
        # Total curl = MCP flexion + PIP flexion. Captures full finger close.
        mcp_flex = _angle_between(palm_fwd, v_mp)
        pip_kink = _angle_between(v_mp, v_pd)
        total = mcp_flex + pip_kink
        # Empirical: extended ≈ 0.2 rad, full curl ≈ 3.0 rad. URDF pip limit 1.48.
        pip_val = np.clip((total - 0.2) / 2.6, 0.0, 1.0) * 1.48
        out[pip_idx] = pip_val
        if abad_idx is not None:
            out[abad_idx] = 0.0

    # Expand to 16 joints matching dex URDF's dof_joint_names layout so the
    # parent retargeter's zip() over get_left_joint_names()/get_right_joint_names()
    # covers pip/dip mimic children too. URDF-observed dex dof order is:
    #   [index_abad, index_pip, index_dip,
    #    middle_pip, middle_dip,
    #    pinky_abad, pinky_pip, pinky_dip,
    #    ring_abad,  ring_pip,  ring_dip,
    #    thumb_roll, thumb_abad, thumb_mcp, thumb_pip, thumb_dip]
    thumb_mcp_val = out[2]
    index_pip_val = out[4]
    middle_pip_val = out[5]
    ring_pip_val = out[7]
    pinky_pip_val = out[9]
    expanded = np.array([
        out[3],                 # index_abad
        index_pip_val,          # index_pip
        1.097 * index_pip_val,  # index_dip (mimic)
        middle_pip_val,         # middle_pip
        1.097 * middle_pip_val, # middle_dip (mimic)
        out[8],                 # pinky_abad
        pinky_pip_val,          # pinky_pip
        1.097 * pinky_pip_val,  # pinky_dip (mimic)
        out[6],                 # ring_abad
        ring_pip_val,           # ring_pip
        1.097 * ring_pip_val,   # ring_dip (mimic)
        out[0],                 # thumb_roll
        out[1],                 # thumb_abad
        thumb_mcp_val,          # thumb_mcp
        1.33 * thumb_mcp_val,   # thumb_pip (mimic)
        1.3  * thumb_mcp_val,   # thumb_dip (mimic)
    ], dtype=np.float32)
    return expanded
