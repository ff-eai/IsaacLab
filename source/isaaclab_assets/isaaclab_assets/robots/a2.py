# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the A2 bipedal humanoid (GenieSim).

The following configuration parameters are available:

* :obj:`A2_CFG`: The A2 dual-arm bipedal humanoid with dexterous hands.

46 revolute DOF:
    - 12 legs (hip_roll, hip_yaw, hip_pitch, tarsus, toe_pitch, toe_roll × 2)
    - 8 arms (4 per arm)
    - 24 hands (12 per hand: thumb swing + 3 thumb + 2 × index/middle/ring/pinky)
    - 2 head
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

A2_ASSETS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))),
    "assets",
    "A2",
)
"""Local asset dir: /home/wagner/code/IsaacLab/assets/A2/"""


def _a2_actuators() -> dict:
    # Full arm regex covers all 7 joints per side (idx13-19 left, idx20-26 right).
    # Arms: stiffness=3000 / damping=40 — dropped damping from 100 (overdamped,
    # sluggish under hand-tracking) to 40 (closer to critical, ~3× faster step
    # response). Matches the OmniHand variant's snappier PD profile while
    # keeping stiffness at the s6-tuned value.
    return {
        "legs": ImplicitActuatorCfg(
            joint_names_expr=["idx0[1-9]_.*", "idx1[0-2]_.*"],
            velocity_limit_sim=None, effort_limit_sim=None, stiffness=None, damping=None,
        ),
        "left_arm": ImplicitActuatorCfg(
            joint_names_expr=["idx1[3-9]_left_arm_joint.*"],
            velocity_limit_sim=None, effort_limit_sim=None, stiffness=3000.0, damping=40.0,
        ),
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=["idx2[0-6]_right_arm_joint.*"],
            velocity_limit_sim=None, effort_limit_sim=None, stiffness=3000.0, damping=40.0,
        ),
        # Hand PD: stiffness 50 holds grip force; damping 4 (was 2) keeps the
        # joint catching up to the higher-rate retargeter output without ringing
        # after low_pass_alpha was raised 0.2→0.5 in the dex yamls.
        "left_hand": ImplicitActuatorCfg(
            joint_names_expr=["L_.*_joint"],
            velocity_limit_sim=None, effort_limit_sim=None, stiffness=50.0, damping=4.0,
        ),
        "right_hand": ImplicitActuatorCfg(
            joint_names_expr=["R_.*_joint"],
            velocity_limit_sim=None, effort_limit_sim=None, stiffness=50.0, damping=4.0,
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["idx2[78]_head_joint.*"],
            velocity_limit_sim=None, effort_limit_sim=None, stiffness=None, damping=None,
        ),
    }


A2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(A2_ASSETS_DIR, "A2_converted.usd"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        activate_contact_sensors=False,  # enable selectively on fingertip links in the env cfg
    ),
    init_state=ArticulationCfg.InitialStateCfg(joint_pos={".*": 0.0}, pos=(0.0, 0.0, 1.05)),
    actuators=_a2_actuators(),
)
"""A2 humanoid — original USD with mimic joints (may hang PhysX)."""


A2_NOMIMIC_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(A2_ASSETS_DIR, "A2_nomimic.usd"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        activate_contact_sensors=False,  # enable selectively on fingertip links in the env cfg
    ),
    init_state=ArticulationCfg.InitialStateCfg(joint_pos={".*": 0.0}, pos=(0.0, 0.0, 1.05)),
    actuators=_a2_actuators(),
)
"""A2 humanoid — mimic joints expanded to independent joints (PhysX-stable).

Use this CFG for task envs. The controller is responsible for setting mimic-child
joints to `multiplier * driver` each step (thumb_2=0.40×thumb_1, thumb_3=0.60×thumb_1,
and each finger_2_joint = 1.0×finger_1_joint)."""


A2_OMNIHAND_ASSETS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))),
    "assets",
    "A2_omnihand",
)

def _a2_omnihand_actuators() -> dict:
    """High-PD actuators matching GR1T2's pick-place config — snappier response
    for VR teleop. Arms: stiffness=4400 / damping=40 (vs stock A2's 3000/100 which
    feels sluggish). Hands: USD defaults (None) — OmniHand has tighter joints and
    the baseline defaults feel tighter than the custom 50/2 A2 s6_hand setting.
    """
    return {
        "legs": ImplicitActuatorCfg(
            joint_names_expr=["idx0[1-9]_.*", "idx1[0-2]_.*"],
            velocity_limit_sim=None, effort_limit_sim=None, stiffness=None, damping=None,
        ),
        "left_arm": ImplicitActuatorCfg(
            joint_names_expr=["idx1[3-9]_left_arm_joint.*"],
            velocity_limit_sim=None, effort_limit_sim=None,
            stiffness=4400.0, damping=40.0, armature=0.01,
        ),
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=["idx2[0-6]_right_arm_joint.*"],
            velocity_limit_sim=None, effort_limit_sim=None,
            stiffness=4400.0, damping=40.0, armature=0.01,
        ),
        "left_hand": ImplicitActuatorCfg(
            joint_names_expr=["L_.*_joint"],
            velocity_limit_sim=None, effort_limit_sim=None,
            stiffness=None, damping=None,
        ),
        "right_hand": ImplicitActuatorCfg(
            joint_names_expr=["R_.*_joint"],
            velocity_limit_sim=None, effort_limit_sim=None,
            stiffness=None, damping=None,
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["idx2[78]_head_joint.*"],
            velocity_limit_sim=None, effort_limit_sim=None, stiffness=None, damping=None,
        ),
    }


A2_OMNIHAND_NOMIMIC_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(A2_OMNIHAND_ASSETS_DIR, "A2_omnihand_nomimic.usd"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(joint_pos={".*": 0.0}, pos=(0.0, 0.0, 1.05)),
    actuators=_a2_omnihand_actuators(),
)
"""A2 humanoid with OmniHand T2 hands — mimic joints expanded.

Hand joint count per side (10 driven):
    thumb_roll, thumb_abad, thumb_mcp
    index_abad, index_pip
    middle_pip                         (middle_abad is FIXED)
    ring_abad,  ring_pip
    pinky_abad, pinky_pip

Controller must set mimic children each step (URDF ratios):
    thumb_pip = 1.33 × thumb_mcp
    thumb_dip = 1.3  × thumb_mcp
    {index,middle,ring,pinky}_dip = 1.097 × {same}_pip"""
