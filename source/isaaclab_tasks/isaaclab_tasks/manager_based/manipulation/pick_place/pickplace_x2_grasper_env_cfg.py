# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Env for X2 humanoid with simple grasper hands: place can into tray.

This is a sibling of `pickplace_x2_env_cfg.py` but swaps the OmniHand for a small
two-finger grasper (URDF in ~/Downloads/robot_description) and exposes a binary
open/close gripper action.
"""

from __future__ import annotations

import tempfile

import torch
from pink.tasks import DampingTask, FrameTask

import carb

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.pink_ik import NullSpacePostureTask, PinkIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.sensors import CameraCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from . import mdp
from .pickplace_x2_env_cfg import _reset_object_left_or_right_of_tray

from isaaclab_assets.robots.x2_grasper import (  # isort: skip
    X2_GRASPER_CFG,
    X2_GRASPER_MESH_PATH,
    X2_GRASPER_URDF_PATH,
)

from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import OpenXRDeviceCfg, XrCfg
from isaaclab.devices.openxr.retargeters.humanoid.agibot.x2_grasper_keyboard import X2GrasperKeyboardCfg
from isaaclab.devices.openxr.retargeters.humanoid.agibot.x2_grasper_retargeter import X2GrasperRetargeterCfg


# --- Tunables ---------------------------------------------------------------
PINK_IK_FRAME_PREFIX = ""
LEFT_EEF_LINK = "left_wrist_roll_link"
RIGHT_EEF_LINK = "right_wrist_roll_link"


CAN_USD = "/home/wagner/code/IsaacLab/assets/robotwin/071_can_base0.usd"
TRAY_USD = "/home/wagner/code/IsaacLab/assets/robotwin/008_tray_base0.usd"

# Real robot intrinsics from `/aima/hal/sensor/rgbd_head_front/rgb_camera_info` (Orbbec Gemini335 RGB).
# Distortion (plumb_bob) is not modeled in Isaac pinhole rendering; K matrix only.
_RGBD_HEAD_FRONT_WIDTH = 1280
_RGBD_HEAD_FRONT_HEIGHT = 720
_RGBD_HEAD_FRONT_INTRINSICS = [
    688.6425170898438,
    0.0,
    642.0774536132812,
    0.0,
    688.736572265625,
    360.6650085449219,
    0.0,
    0.0,
    1.0,
]

# SenYun SDS23NNS1 stereo pair — `/aima/hal/sensor/stereo_head_front_{left,right}/camera_info`.
# Fisheye distortion (4 coeffs) is not modeled; K matrix only. Baseline from left P[3]/fx ≈ 61 mm.
_STEREO_HEAD_WIDTH = 2064
_STEREO_HEAD_HEIGHT = 1552
_STEREO_HEAD_BASELINE_M = 41.56786639220563 / 681.304426892005  # ≈ 0.061 m
_STEREO_HEAD_FRONT_LEFT_INTRINSICS = [
    684.17192619,
    0.0,
    1026.9281357,
    0.0,
    684.4766955729,
    774.884008384,
    0.0,
    0.0,
    1.0,
]
_STEREO_HEAD_FRONT_RIGHT_INTRINSICS = [
    682.4560723158,
    0.0,
    1029.5072118988,
    0.0,
    682.4665814761,
    776.3256584051,
    0.0,
    0.0,
    1.0,
]
# URDF stereo_head_front joint center on head_pitch_link (same mount as pickplace_x2_env_cfg).
_STEREO_HEAD_CENTER_POS = (0.20, 0.02978, 0.05000)
_STEREO_HEAD_MOUNT_ROT = (0.49920, -0.49920, 0.50080, -0.50080)
_STEREO_HALF_BASELINE = _STEREO_HEAD_BASELINE_M / 2.0

_LEFT_ARM_JOINTS = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
]
_RIGHT_GRIPPER_JOINTS = ["R_hand_narrow1_joint", "R_hand_wide1_joint"]
_LEFT_GRIPPER_JOINTS = ["L_hand_narrow1_joint", "L_hand_wide1_joint"]

_RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
]
_WAIST_JOINTS = ["waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint"]

# Standing height from URDF FK (neutral legs): lowest ankle link is ~0.602 m below root.
# Gravity is disabled on this robot, so spawn z must place feet on the ground plane.
# A few mm of clearance avoids PhysX depenetration pushing the root upward on reset.
_X2_FOOT_OFFSET_FROM_ROOT = 0.602
_X2_PELVIS_Z_STANDING = _X2_FOOT_OFFSET_FROM_ROOT + 0.005
_TABLETOP_Z = 0.82  # table height independent of pelvis (+20 cm vs prior 0.62)
_TABLE_SLAB_THICKNESS = 0.05
_TABLE_SLAB_CENTER_Z = _TABLETOP_Z - _TABLE_SLAB_THICKNESS / 2
# Keep leg bottoms on z=0: leg top touches tabletop bottom (z=tabletop - thickness).
_TABLE_LEG_HEIGHT = _TABLETOP_Z - _TABLE_SLAB_THICKNESS
_TABLE_LEG_CENTER_Z = _TABLE_LEG_HEIGHT / 2

# Ready-pose joint angles (must stay in sync with `ObjectTableSceneCfg.robot.init_state`).
_READY_JOINT_POS = {
    "left_shoulder_pitch_joint": 0.0,
    "left_shoulder_roll_joint": 0.5,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": -1.0,
    "left_wrist_yaw_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_roll_joint": 0.0,
    "right_shoulder_pitch_joint": 0.0,
    "right_shoulder_roll_joint": -0.5,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": -1.0,
    "right_wrist_yaw_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_roll_joint": 0.0,
}
_ROBOT_ROOT_POS = (0.0, 0.0, _X2_PELVIS_Z_STANDING)
_ROBOT_ROOT_ROT_WXYZ = (0.7071, 0.0, 0.0, 0.7071)


def _fk_wrist_pose_world(link_name: str) -> tuple[float, ...]:
    """World-frame wrist pose (x,y,z,qw,qx,qy,qz) for the ready arm pose."""
    import pinocchio as pin
    import numpy as np
    from scipy.spatial.transform import Rotation as R

    model = pin.buildModelFromUrdf(X2_GRASPER_URDF_PATH)
    data = model.createData()
    q = pin.neutral(model)
    for jid in range(1, model.njoints):
        jname = model.names[jid]
        if jname not in _READY_JOINT_POS:
            continue
        q[model.joints[jid].idx_q] = _READY_JOINT_POS[jname]

    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    fid = model.getFrameId(link_name)
    p_local = np.asarray(data.oMf[fid].translation, dtype=np.float64)
    r_local = np.asarray(data.oMf[fid].rotation, dtype=np.float64)

    root_pos = np.asarray(_ROBOT_ROOT_POS, dtype=np.float64)
    w, x, y, z = _ROBOT_ROOT_ROT_WXYZ
    r_root = R.from_quat([x, y, z, w])
    p_world = r_root.apply(p_local) + root_pos
    r_world = r_root.as_matrix() @ r_local
    qx, qy, qz, qw = R.from_matrix(r_world).as_quat()
    return (float(p_world[0]), float(p_world[1]), float(p_world[2]), float(qw), float(qx), float(qy), float(qz))


def _build_idle_action() -> torch.Tensor:
    left_wrist = _fk_wrist_pose_world(LEFT_EEF_LINK)
    right_wrist = _fk_wrist_pose_world(RIGHT_EEF_LINK)
    return torch.tensor([*left_wrist, *right_wrist, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)


##
# Scene
##
@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Scene: X2 humanoid + packing table + can + tray."""

    ground = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=sim_utils.GroundPlaneCfg())

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    packing_table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.55, _TABLE_SLAB_CENTER_Z), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.MeshCuboidCfg(
            size=(1.2, 0.7, _TABLE_SLAB_THICKNESS),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.95, 0.95), roughness=0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    table_leg_0 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/TableLeg0",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.55 + 0.30, _TABLE_LEG_CENTER_Z)),
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.1, 0.1, _TABLE_LEG_HEIGHT),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.95, 0.95), roughness=0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )
    table_leg_1 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/TableLeg1",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.55 - 0.30, _TABLE_LEG_CENTER_Z)),
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.1, 0.1, _TABLE_LEG_HEIGHT),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.95, 0.95), roughness=0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )
    table_leg_2 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/TableLeg2",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.55, 0.55 + 0.30, _TABLE_LEG_CENTER_Z)),
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.1, 0.1, _TABLE_LEG_HEIGHT),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.95, 0.95), roughness=0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )
    table_leg_3 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/TableLeg3",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.55, 0.55 - 0.30, _TABLE_LEG_CENTER_Z)),
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.1, 0.1, _TABLE_LEG_HEIGHT),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.95, 0.95), roughness=0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    # Place robot on the ground, facing the table, 20 cm from table front edge.
    # Table center is at y=0.55 with width=0.7 → front edge at y=0.20.
    # Distance 0.20 m → robot root at y=0.00.
    robot: ArticulationCfg = X2_GRASPER_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.00, _X2_PELVIS_Z_STANDING),
            rot=(0.7071, 0.0, 0.0, 0.7071),  # face +y (towards table)
            joint_pos={
                # Arms: ready pose (copied from X2 omnihand task).
                "left_shoulder_pitch_joint": 0.0,
                "left_shoulder_roll_joint": 0.5,
                "left_shoulder_yaw_joint": 0.0,
                "left_elbow_joint": -1.0,
                "left_wrist_yaw_joint": 0.0,
                "left_wrist_pitch_joint": 0.0,
                "left_wrist_roll_joint": 0.0,
                "right_shoulder_pitch_joint": 0.0,
                "right_shoulder_roll_joint": -0.5,
                "right_shoulder_yaw_joint": 0.0,
                "right_elbow_joint": -1.0,
                "right_wrist_yaw_joint": 0.0,
                "right_wrist_pitch_joint": 0.0,
                "right_wrist_roll_joint": 0.0,
                # Legs/waist/head neutral.
                ".*_hip_pitch_joint": 0.0,
                ".*_hip_roll_joint": 0.0,
                ".*_hip_yaw_joint": 0.0,
                ".*_knee_joint": 0.0,
                ".*_ankle_pitch_joint": 0.0,
                ".*_ankle_roll_joint": 0.0,
                "waist_.*_joint": 0.0,
                "head_.*_joint": 0.0,
                # Graspers open.
                "L_hand_.*_joint": 0.0,
                "R_hand_.*_joint": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
    )

    # Can starts upright on the table surface.
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.22, 0.40, _TABLETOP_Z + 0.10), rot=(0.707, 0.707, 0.0, 0.0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=CAN_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=5.0),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
    )

    # Tray sits flat on the table (kinematic so it doesn't drift).
    tray: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Tray",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.55, _TABLETOP_Z), rot=(0.707, 0.707, 0.0, 0.0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=TRAY_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, kinematic_enabled=True, max_depenetration_velocity=5.0
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
    )

    # rgbd_head_front — Orbbec Gemini335 RGB-D (frame_id: rgbd_head_front, 1280×720).
    # URDF fixed joint on head_pitch_link; prim uses `_cam` suffix to avoid clashing with URDF link name.
    rgbd_head_front = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/head_pitch_link/rgbd_head_front_cam",
        update_period=0.05,
        height=_RGBD_HEAD_FRONT_HEIGHT,
        width=_RGBD_HEAD_FRONT_WIDTH,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=_RGBD_HEAD_FRONT_INTRINSICS,
            width=_RGBD_HEAD_FRONT_WIDTH,
            height=_RGBD_HEAD_FRONT_HEIGHT,
            clipping_range=(0.05, 20.0),
        ),
        offset=CameraCfg.OffsetCfg(
            # URDF rgbd_head_front joint xyz=(0.05761,-0.01118,-0.04837) rpy=(2.2689,0,1.5708);
            # pushed +14 cm forward so frustum clears the head shell (same as pickplace_x2_env_cfg).
            pos=(0.20, -0.01118, -0.04837),
            rot=(0.29884, 0.64085, 0.64085, 0.29885),
            convention="world",
        ),
    )

    # stereo_head_front_left / _right — SenYun SDS23NNS1 (2064×1552, frame_ids stereo_head_front / stereo_head_front_right).
    stereo_head_front_left = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/head_pitch_link/stereo_head_front_left_cam",
        update_period=0.05,
        height=_STEREO_HEAD_HEIGHT,
        width=_STEREO_HEAD_WIDTH,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=_STEREO_HEAD_FRONT_LEFT_INTRINSICS,
            width=_STEREO_HEAD_WIDTH,
            height=_STEREO_HEAD_HEIGHT,
            clipping_range=(0.05, 20.0),
        ),
        offset=CameraCfg.OffsetCfg(
            # URDF stereo center ± half baseline along head +y (robot-left).
            pos=(
                _STEREO_HEAD_CENTER_POS[0],
                _STEREO_HEAD_CENTER_POS[1] + _STEREO_HALF_BASELINE,
                _STEREO_HEAD_CENTER_POS[2],
            ),
            rot=_STEREO_HEAD_MOUNT_ROT,
            convention="world",
        ),
    )
    stereo_head_front_right = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/head_pitch_link/stereo_head_front_right_cam",
        update_period=0.05,
        height=_STEREO_HEAD_HEIGHT,
        width=_STEREO_HEAD_WIDTH,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=_STEREO_HEAD_FRONT_RIGHT_INTRINSICS,
            width=_STEREO_HEAD_WIDTH,
            height=_STEREO_HEAD_HEIGHT,
            clipping_range=(0.05, 20.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(
                _STEREO_HEAD_CENTER_POS[0],
                _STEREO_HEAD_CENTER_POS[1] - _STEREO_HALF_BASELINE,
                _STEREO_HEAD_CENTER_POS[2],
            ),
            rot=_STEREO_HEAD_MOUNT_ROT,
            convention="world",
        ),
    )


##
# Actions
##
@configclass
class ActionsCfg:
    """Action: Pink IK for both arms + binary gripper open/close."""

    upper_body_ik = PinkInverseKinematicsActionCfg(
        pink_controlled_joint_names=_LEFT_ARM_JOINTS + _RIGHT_ARM_JOINTS,
        hand_joint_names=_LEFT_GRIPPER_JOINTS + _RIGHT_GRIPPER_JOINTS,
        target_eef_link_names={"left_wrist": LEFT_EEF_LINK, "right_wrist": RIGHT_EEF_LINK},
        asset_name="robot",
        controller=PinkIKControllerCfg(
            articulation_name="robot",
            base_link_name="base_link",
            num_hand_joints=len(_LEFT_GRIPPER_JOINTS) + len(_RIGHT_GRIPPER_JOINTS),
            # qpsolvers 4.x + older daqp wheels reject ``primal_start``; proxqp works.
            solver="proxqp",
            show_ik_warnings=False,
            fail_on_joint_limit_violation=False,
            variable_input_tasks=[
                FrameTask(
                    f"{PINK_IK_FRAME_PREFIX}{LEFT_EEF_LINK}",
                    position_cost=8.0,
                    orientation_cost=1.0,
                    lm_damping=12,
                    gain=0.5,
                ),
                FrameTask(
                    f"{PINK_IK_FRAME_PREFIX}{RIGHT_EEF_LINK}",
                    position_cost=8.0,
                    orientation_cost=1.0,
                    lm_damping=12,
                    gain=0.5,
                ),
                DampingTask(cost=0.5),
                NullSpacePostureTask(
                    cost=0.5,
                    lm_damping=1,
                    controlled_frames=[
                        f"{PINK_IK_FRAME_PREFIX}{LEFT_EEF_LINK}",
                        f"{PINK_IK_FRAME_PREFIX}{RIGHT_EEF_LINK}",
                    ],
                    controlled_joints=_LEFT_ARM_JOINTS[:4] + _RIGHT_ARM_JOINTS[:4] + _WAIST_JOINTS,
                ),
            ],
            fixed_input_tasks=[],
            xr_enabled=bool(carb.settings.get_settings().get("/app/xr/enabled")),
        ),
    )


##
# Observations
##
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action)
        robot_joint_pos = ObsTerm(func=base_mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_root_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_root_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("robot")})
        object_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("object")})
        object_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("object")})
        tray_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("tray")})
        tray_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("tray")})

        left_eef_pos = ObsTerm(func=mdp.get_eef_pos, params={"link_name": LEFT_EEF_LINK})
        left_eef_quat = ObsTerm(func=mdp.get_eef_quat, params={"link_name": LEFT_EEF_LINK})
        right_eef_pos = ObsTerm(func=mdp.get_eef_pos, params={"link_name": RIGHT_EEF_LINK})
        right_eef_quat = ObsTerm(func=mdp.get_eef_quat, params={"link_name": RIGHT_EEF_LINK})

        object = ObsTerm(
            func=mdp.object_obs,
            params={"left_eef_link_name": LEFT_EEF_LINK, "right_eef_link_name": RIGHT_EEF_LINK},
        )

        rgbd_head_front_rgb = ObsTerm(
            func=base_mdp.image,
            params={"sensor_cfg": SceneEntityCfg("rgbd_head_front"), "data_type": "rgb", "normalize": False},
        )
        rgbd_head_front_depth = ObsTerm(
            func=base_mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("rgbd_head_front"),
                "data_type": "distance_to_image_plane",
                "normalize": False,
            },
        )
        stereo_head_front_left_rgb = ObsTerm(
            func=base_mdp.image,
            params={"sensor_cfg": SceneEntityCfg("stereo_head_front_left"), "data_type": "rgb", "normalize": False},
        )
        stereo_head_front_right_rgb = ObsTerm(
            func=base_mdp.image,
            params={"sensor_cfg": SceneEntityCfg("stereo_head_front_right"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


##
# Terminations
##
@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.35, "asset_cfg": SceneEntityCfg("object")},
    )

    success = DoneTerm(
        func=mdp.place_after_grasp,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "object_a_cfg": SceneEntityCfg("object"),
            "object_b_cfg": SceneEntityCfg("tray"),
            "xy_threshold": 0.10,
            "height_diff": 0.05,
            "height_threshold": 0.05,
            "grasp_joint_names": ("R_hand_narrow1_joint", "R_hand_wide1_joint"),
            "grasp_closed_threshold": 0.4,
            "grasp_proximity": 0.15,
            "hand_body_name": RIGHT_EEF_LINK,
        },
    )


##
# Events
##
@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_object = EventTerm(
        func=_reset_object_left_or_right_of_tray,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("object"), "offset_m": 0.22, "jitter_m": 0.05},
    )


##
# Env cfg
##
@configclass
class PickPlaceX2GrasperEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for X2 (with graspers) place-can-into-tray environment."""

    # Put the robot + table in-frame by default (env origin frame).
    viewer: ViewerCfg = ViewerCfg(
        eye=(2.2, 2.2, 1.5), lookat=(0.0, 0.55, _TABLETOP_Z), origin_type="env"
    )

    xr: XrCfg = XrCfg(anchor_pos=(0.0, 0.0, 0.0), anchor_rot=(1.0, 0.0, 0.0, 0.0))
    NUM_OPENXR_HAND_JOINTS = 26

    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events = EventCfg()

    commands = None
    rewards = None
    curriculum = None

    temp_urdf_dir = tempfile.gettempdir()

    gripper_joint_names = ["R_hand_narrow1_joint", "R_hand_wide1_joint"]
    gripper_open_val = 0.0
    gripper_threshold = 0.1

    # Idle action: left wrist (7) + right wrist (7) + 4 grasper joints (open).
    # Wrist targets are computed from URDF FK so Pink IK and keyboard teleop start
    # aligned with the rendered ready pose (avoids silent IK failure on first step).
    idle_action: torch.Tensor = _build_idle_action()

    def __post_init__(self):
        self.decimation = 6
        self.episode_length_s = 20.0
        self.sim.dt = 1 / 120
        self.sim.render_interval = 2

        # Use the generated merged URDF for Pink IK.
        self.actions.upper_body_ik.controller.urdf_path = X2_GRASPER_URDF_PATH
        # X2 body meshes are relative (meshes/x2/...); grasper meshes use file:// URIs in the URDF.
        self.actions.upper_body_ik.controller.mesh_path = X2_GRASPER_MESH_PATH

        idle = self.idle_action.tolist()
        left_idle_pose = tuple(idle[0:7])
        right_idle_pose = tuple(idle[7:14])

        left_fixed_rpy = (0.0, 0.0, 0.0)
        right_fixed_rpy = (0.0, 0.0, 3.1415927)
        left_shoulder_pos = (-0.2, 0.05, _TABLETOP_Z + 0.45)
        right_shoulder_pos = (0.2, 0.05, _TABLETOP_Z + 0.45)
        grasper_joint_names = _LEFT_GRIPPER_JOINTS + _RIGHT_GRIPPER_JOINTS

        self.teleop_devices = DevicesCfg(
            devices={
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        X2GrasperRetargeterCfg(
                            enable_visualization=True,
                            num_open_xr_hand_joints=2 * self.NUM_OPENXR_HAND_JOINTS,
                            sim_device=self.sim.device,
                            grasper_joint_names=grasper_joint_names,
                            left_fixed_rpy=left_fixed_rpy,
                            right_fixed_rpy=right_fixed_rpy,
                            left_shoulder_pos=left_shoulder_pos,
                            right_shoulder_pos=right_shoulder_pos,
                            max_reach=0.7,
                            debug_every=30,
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
                "keyboard": X2GrasperKeyboardCfg(
                    sim_device=self.sim.device,
                    pos_sensitivity=0.4,
                    rot_sensitivity=0.8,
                    left_wrist_idle=left_idle_pose,
                    right_wrist_start=right_idle_pose,
                    dt=1.0 / 30.0,
                ),
            }
        )

