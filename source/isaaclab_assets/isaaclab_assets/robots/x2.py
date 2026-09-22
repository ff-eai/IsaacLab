# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the X2 bipedal humanoid with OmniHand T2 hands.

X2 ultra body has:
    - 12 leg joints (hip pitch/roll/yaw, knee, ankle pitch/roll × 2)
    - 3 waist joints (yaw, pitch, roll)
    - 14 arm joints (shoulder pitch/roll/yaw, elbow, wrist yaw/pitch/roll × 2)
    - 2 head joints (yaw, pitch)

Plus 32 OmniHand T2 joints (16 per hand). After URDF merge_fixed_joints=True the
fixed L_palm_joint/R_palm_joint between wrist_roll_link and palm collapses, so
Pink IK targets `left_wrist_roll_link` / `right_wrist_roll_link` directly.

Hand joint count per side (10 driven + 6 mimic children):
    thumb_roll, thumb_abad, thumb_mcp        (driven)
    index_abad, index_pip                    (driven)
    middle_pip                               (driven; middle_abad is FIXED)
    ring_abad,  ring_pip                     (driven)
    pinky_abad, pinky_pip                    (driven)
    thumb_pip, thumb_dip                     (mimic of thumb_mcp)
    {index,middle,ring,pinky}_dip            (mimic of *_pip)
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

X2_OMNIHAND_ASSETS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))),
    "assets",
    "X2_omnihand",
)
"""Local asset dir: /home/wagner/code/IsaacLab/assets/X2_omnihand/"""


def _x2_omnihand_actuators() -> dict:
    """Mirror A2 OmniHand tuning: stiff arms (4400/40) for snappy VR teleop tracking,
    USD-default hands. Legs/waist/head left at USD defaults — gravity is disabled
    on this robot so they hold whatever init pose is set.
    """
    return {
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"],
            velocity_limit_sim=None, effort_limit_sim=None, stiffness=None, damping=None,
        ),
        "waist": ImplicitActuatorCfg(
            # High-PD waist so gravity / reaction torques from arm motion can't
            # drift the waist away from its commanded zero. The IK doesn't
            # control the waist (it's not in pink_controlled_joint_names) so
            # the PD target is the InitialStateCfg value (0 for all 3 axes).
            joint_names_expr=["waist_.*_joint"],
            velocity_limit_sim=None, effort_limit_sim=None,
            stiffness=4400.0, damping=40.0, armature=0.01,
        ),
        "left_arm": ImplicitActuatorCfg(
            joint_names_expr=["left_shoulder_.*", "left_elbow_joint", "left_wrist_.*"],
            velocity_limit_sim=None, effort_limit_sim=None,
            stiffness=4400.0, damping=40.0, armature=0.01,
        ),
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=["right_shoulder_.*", "right_elbow_joint", "right_wrist_.*"],
            velocity_limit_sim=None, effort_limit_sim=None,
            stiffness=4400.0, damping=40.0, armature=0.01,
        ),
        # OmniHand finger PD copied from A2's working config (a2.py): stiffness=50,
        # damping=2 gives the fingers enough drive force to hold the can. Without
        # these, USD-default stiffness is too soft and the can slips out of the
        # closed grip even when the retargeted joint targets are correct.
        "left_hand": ImplicitActuatorCfg(
            joint_names_expr=["L_.*_joint"],
            velocity_limit_sim=None, effort_limit_sim=None,
            stiffness=50.0, damping=2.0,
        ),
        "right_hand": ImplicitActuatorCfg(
            joint_names_expr=["R_.*_joint"],
            velocity_limit_sim=None, effort_limit_sim=None,
            stiffness=50.0, damping=2.0,
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_.*_joint"],
            velocity_limit_sim=None, effort_limit_sim=None, stiffness=None, damping=None,
        ),
    }


X2_OMNIHAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(X2_OMNIHAND_ASSETS_DIR, "X2_omnihand.usd"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(joint_pos={".*": 0.0}, pos=(0.0, 0.0, 1.05)),
    actuators=_x2_omnihand_actuators(),
)
"""X2 humanoid with OmniHand T2 hands — mimic joints expanded.

Controller must set mimic children each step (URDF ratios; copied from A2 omnihand):
    thumb_pip = 1.33  × thumb_mcp
    thumb_dip = 1.30  × thumb_mcp
    {index,middle,ring,pinky}_dip = 1.097 × {same}_pip"""
