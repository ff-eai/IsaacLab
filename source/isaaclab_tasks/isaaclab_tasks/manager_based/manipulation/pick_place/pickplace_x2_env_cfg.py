# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Env for X2 humanoid with OmniHand T2 hands: place can into tray via VR teleop.

Forked from pickplace_a2_omnihand_env_cfg.py. Uses X2_OMNIHAND_CFG and the
OmniHand dex-retargeting path (DexPilot, 10 DoF per hand driven; pip/dip joints
populated via mimic ratios from the URDF in the retargeter output).

X2 has a 7-DOF arm (shoulder pitch/roll/yaw + elbow + wrist yaw/pitch/roll) and
a 3-DOF waist (yaw/pitch/roll), unlike A2's 7-DOF arm + 1-DOF waist (yaw only).

Known tuning points (will fail/misbehave until validated on your rig):
  * Camera mount links (head_pitch_link, torso_link) and offsets are copied from
    A2 — they will likely need retuning for X2's body geometry.
  * Init pose joint values are mid-range guesses; tune after first sim run.
  * Palm orientation (L_palm_joint / R_palm_joint rpy in the merged URDF) is
    (0,0,0) — A2 needed asymmetric ±π/2 Z; X2 may need similar tuning.
"""

from __future__ import annotations

import tempfile

import torch
from pink.tasks import DampingTask, FrameTask

import carb

import isaaclab.controllers.utils as ControllerUtils
import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.pink_ik import NullSpacePostureTask, PinkIKControllerCfg
from isaaclab.sensors import CameraCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import OpenXRDeviceCfg, XrCfg
from isaaclab.devices.openxr.retargeters.humanoid.agibot.a2_omnihand_retargeter import A2OmniHandRetargeterCfg
from isaaclab.devices.openxr.retargeters.humanoid.agibot.a2_omnihand_keyboard import A2OmniHandKeyboardCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from . import mdp
from isaaclab_tasks.manager_based.manipulation.place import mdp as place_mdp

from isaaclab_assets.robots.x2 import X2_OMNIHAND_CFG  # isort: skip


def _hide_packing_table_crate(env, env_ids):
    """Walk each env's PackingTable prim tree and hide any sub-prim whose name
    mentions a crate/tray/bin mesh so the table surface stays clean. Runs once
    as a startup event."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    # Match any sub-prim whose name contains these substrings — covers the plastic
    # crate, the inner tray mesh, and the packaged goods that ship in the USD.
    hide_patterns = ("Crate", "Tray", "Bin", "Box_", "Package")
    env_indices = range(env.num_envs) if env_ids is None else env_ids.cpu().tolist()
    hidden = 0
    for env_i in env_indices:
        root = stage.GetPrimAtPath(f"/World/envs/env_{env_i}/PackingTable")
        if not root.IsValid():
            continue
        for prim in root.GetAllChildren():
            for sub in [prim, *prim.GetAllChildren()]:
                name = sub.GetName()
                if any(p in name for p in hide_patterns):
                    img = UsdGeom.Imageable(sub)
                    if img:
                        img.MakeInvisible()
                        hidden += 1
    print(f"[hide_crate] hid {hidden} sub-prims of PackingTable matching {hide_patterns}")


def _reset_object_left_or_right_of_tray(
    env,
    env_ids,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    offset_m: float = 0.22,
    jitter_m: float = 0.05,
):
    """Place the can on the RIGHT side of the tray:
        x = env_origin_x + offset_m + U[-jitter_m, +jitter_m]
        y = env_origin_y + jitter_m * U[-1, 1]   (y also jittered ±jitter_m)
        z = default
    Velocity zeroed."""
    asset = env.scene[asset_cfg.name]
    root_states = asset.data.default_root_state[env_ids].clone()
    n = len(env_ids)
    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids]
    jitter_xy = (torch.rand((n, 2), device=asset.device, dtype=root_states.dtype) * 2.0 - 1.0) * jitter_m
    positions[:, 0] = env.scene.env_origins[env_ids][:, 0] + offset_m + jitter_xy[:, 0]
    positions[:, 1] = positions[:, 1] + jitter_xy[:, 1]
    orientations = root_states[:, 3:7]
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.zeros((n, 6), device=asset.device), env_ids=env_ids)


# --- Tunables ---------------------------------------------------------------
# After URDF→USD with merge_fixed_joints=True, the fixed L_palm_joint /
# R_palm_joint between *_wrist_roll_link and L/R_palm collapses into the parent,
# so the IK target frame is *_wrist_roll_link.
PINK_IK_FRAME_PREFIX = ""
LEFT_EEF_LINK = "left_wrist_roll_link"
RIGHT_EEF_LINK = "right_wrist_roll_link"

# RoboTwin 071_can (matches place_can_basket task) — converted from
# /home/wagner/code/RoboTwin/assets/objects/071_can/visual/base0.glb via
# scripts/demos/convert_robotwin_assets.py.
CAN_USD = "/home/wagner/code/IsaacLab/assets/robotwin/071_can_base0.usd"
# RoboTwin 008_tray (matches place_can_basket task).
TRAY_USD = "/home/wagner/code/IsaacLab/assets/robotwin/008_tray_base0.usd"

# Per-side arm joint names — order is shoulder/elbow first, wrist last.
# Wrist comes BEFORE the OmniHand attachment in the kinematic chain.
_LEFT_ARM_JOINTS = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint",
]
_RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint",
]
# X2 has a 3-DOF waist; add to pink_controlled_joint_names below if you want IK
# to use it. Default: not controlled (held at zero by USD defaults).
_WAIST_JOINTS = ["waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint"]

# OmniHand T2 joints in the nomimic USD — 16 per side. The first 10 per side are
# the DRIVEN joints the dex retargeter outputs; the remaining pip/dip children
# are populated via URDF mimic multipliers inside the retargeter.
_HAND_JOINTS = [
    # Left: 3 thumb driven + 2 index + 1 middle + 2 ring + 2 pinky = 10 driven,
    #       then 3 thumb mimic (pip,dip) + 4 finger dip = 6 mimic children.
    "L_thumb_roll_joint", "L_thumb_abad_joint", "L_thumb_mcp_joint",
    "L_index_abad_joint", "L_index_pip_joint",
    "L_middle_pip_joint",
    "L_ring_abad_joint", "L_ring_pip_joint",
    "L_pinky_abad_joint", "L_pinky_pip_joint",
    "L_thumb_pip_joint", "L_thumb_dip_joint",
    "L_index_dip_joint", "L_middle_dip_joint", "L_ring_dip_joint", "L_pinky_dip_joint",
    # Right (same layout).
    "R_thumb_roll_joint", "R_thumb_abad_joint", "R_thumb_mcp_joint",
    "R_index_abad_joint", "R_index_pip_joint",
    "R_middle_pip_joint",
    "R_ring_abad_joint", "R_ring_pip_joint",
    "R_pinky_abad_joint", "R_pinky_pip_joint",
    "R_thumb_pip_joint", "R_thumb_dip_joint",
    "R_index_dip_joint", "R_middle_dip_joint", "R_ring_dip_joint", "R_pinky_dip_joint",
]


##
# Scene
##
@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Scene: A2 humanoid + packing table + can + tray."""

    # Procedural table matching RoboTwin's _base_task.create_table (1.2 × 0.7 × 0.05
    # slab at top surface z=1.0, white, kinematic). Ported from
    # /home/wagner/code/RoboTwin/envs/utils/create_actor.py::create_table.
    packing_table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable",
        # tabletop center = top_z - thickness/2 = 1.0 - 0.025 = 0.975
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.55, 0.975), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.MeshCuboidCfg(
            size=(1.2, 0.7, 0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.95, 0.95), roughness=0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    # 4 table legs, 0.1 × 0.1 × 1.0 each, at the corners. Leg top meets tabletop
    # bottom (z=0.95), leg bottom at floor (z≈0). Leg spacing matches create_table:
    # x = ±(length/2 - leg_spacing/2) = ±0.55, y = ±(width/2 - leg_spacing/2) = ±0.30.
    table_leg_0 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/TableLeg0",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.55 + 0.30, 0.475)),
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.1, 0.1, 0.95),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.95, 0.95), roughness=0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )
    table_leg_1 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/TableLeg1",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.55 - 0.30, 0.475)),
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.1, 0.1, 0.95),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.95, 0.95), roughness=0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )
    table_leg_2 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/TableLeg2",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.55, 0.55 + 0.30, 0.475)),
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.1, 0.1, 0.95),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.95, 0.95), roughness=0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )
    table_leg_3 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/TableLeg3",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.55, 0.55 - 0.30, 0.475)),
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.1, 0.1, 0.95),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.95, 0.95), roughness=0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    # RoboTwin 071_can — same asset used in place_can_basket task. Mass 0.05 kg
    # per place_can_basket.py. Quaternion [0.707, 0.707, 0, 0] (wxyz) matches
    # RoboTwin's rand_create_actor call (upright standing orientation).
    # Default pos sits the can to the RIGHT of the tray at startup (the reset
    # event below picks the actual side per-episode and sets x absolutely).
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.22, 0.40, 1.10], rot=[0.707, 0.707, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=CAN_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )

    # RoboTwin 008_tray — same asset used in place_can_basket. Mass 0.85 kg per
    # place_can_basket.py. Kinematic_enabled=True keeps it stable on the table
    # while the can is placed into it (avoids PhysX velocity-on-kinematic warning
    # from reset events as the 'hide_crate' isn't running on this asset).
    # Tray fixed at y=0.40 (15cm closer to the robot than the original 0.55 — gives
    # the OmniHand some reach-margin for picking the can off the tray's flank).
    tray = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Tray",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.0, 0.40, 1.00], rot=[0.707, 0.707, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=TRAY_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.85),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )

    robot: ArticulationCfg = X2_OMNIHAND_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0.05, 0.93),  # 5cm away from table — TODO retune for X2 height
            rot=(0.7071, 0, 0, 0.7071),  # face +y (towards table)
            joint_pos={
                # Arms: ready pose. left_elbow is one-sided [-2.3556, 0]; pre-bend
                # to mid-range so Pink IK has authority in both directions.
                # left_shoulder_roll is one-sided [-0.061, 2.993]; raise shoulder
                # forward so reach is comfortable.
                # Raised arm init: shoulder_roll lifts arm; elbow bends
                # (both elbows are one-sided [-2.3556, 0] so use negative).
                # Reach-extended ready pose: less side-raise + less elbow-bend than
                # before (1.0 → 0.5 rad shoulder_roll; -1.5 → -1.0 rad elbow). This
                # extends the wrist further forward so the IK starts closer to the
                # can position at world y≈0.4 / x=±0.22 and converges faster.
                "left_shoulder_pitch_joint": 0.0,
                "left_shoulder_roll_joint": 0.5,    # raise arm (+29°)
                "left_shoulder_yaw_joint": 0.0,
                "left_elbow_joint": -1.0,           # bend (−57°)
                "left_wrist_yaw_joint": 0.0,
                "left_wrist_pitch_joint": 0.0,
                "left_wrist_roll_joint": 0.0,
                "right_shoulder_pitch_joint": 0.0,
                "right_shoulder_roll_joint": -0.5,  # mirror of left
                "right_shoulder_yaw_joint": 0.0,
                "right_elbow_joint": -1.0,
                "right_wrist_yaw_joint": 0.0,
                "right_wrist_pitch_joint": 0.0,
                "right_wrist_roll_joint": 0,
                # Legs: straight (gravity disabled, so they stay).
                ".*_hip_pitch_joint": 0.0,
                ".*_hip_roll_joint": 0.0,
                ".*_hip_yaw_joint": 0.0,
                ".*_knee_joint": 0.0,
                ".*_ankle_pitch_joint": 0.0,
                ".*_ankle_roll_joint": 0.0,
                # Waist + head neutral.
                "waist_.*_joint": 0.0,
                "head_.*_joint": 0.0,
                "L_.*": 0.0,
                "R_.*": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
    )

    ground = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=GroundPlaneCfg())

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # --- Cameras (X2 sensors per https://x2-aimdk.agibot.com/en/latest/about_agibot_X2/sensor_fov.html)
    # All four head cameras parent to head_pitch_link with the URDF-defined
    # fixed-joint poses (xyz + rpy converted to wxyz). Resolutions are reduced
    # from datasheet to keep sim render cost reasonable; bump up if needed.
    # Focal length back-solved from spec HFOV at horizontal_aperture=20.955:
    #   HFOV=94°  → f≈9.770mm  (Orbbec Gemini335 RGB-D)
    #   HFOV=156° → f≈2.227mm  (SDS23NNS1 stereo / SM3S23NS rear)
    #   HFOV=110° → f≈7.336mm  (SM5M12NJ front interaction)
    #
    # NOTE: The URDF-spec camera positions are tucked tightly against (or just
    # inside) the head mesh shell, so the camera frusta are occluded by the head
    # geometry and the cameras themselves render through it. We push the front
    # trio +14cm forward (+x in head_pitch_link) and the rear cam −12cm back so
    # they sit clearly outside the head shell. This sacrifices ~14cm of strict
    # mounting realism in exchange for unobstructed views during teleop/eval.

    # rgbd_head_front — Orbbec Gemini335 (RGB-D, 94° HFOV, 1280×720 here, depth+rgb).
    # NOTE: prim child name uses `_cam` suffix because the X2 URDF already has
    # links literally named `rgbd_head_front` etc. (camera mount geometry).
    # Without the suffix the camera prim collides with the URDF link prim.
    rgbd_head_front = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/head_pitch_link/rgbd_head_front_cam",
        update_period=0.05,
        height=720,
        width=1280,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=9.77, focus_distance=400.0,
            horizontal_aperture=20.955, clipping_range=(0.05, 20.0),
        ),
        offset=CameraCfg.OffsetCfg(
            # URDF-spec x=0.05761 → 0.20 (pushed +14cm forward of head shell).
            pos=(0.20, -0.01118, -0.04837),
            rot=(0.29884, 0.64085, 0.64085, 0.29885),
            convention="world",
        ),
    )

    # stereo_head_front — SenYun SDS23NNS1 (stereo RGB, 156° HFOV).
    # Sim renders this as a single mono RGB; treat as "left eye" of the stereo pair.
    stereo_head_front = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/head_pitch_link/stereo_head_front_cam",
        update_period=0.05,
        height=768,
        width=1024,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.23, focus_distance=400.0,
            horizontal_aperture=20.955, clipping_range=(0.05, 20.0),
        ),
        offset=CameraCfg.OffsetCfg(
            # URDF-spec x=0.06799 → 0.20 (pushed +14cm forward of head shell).
            pos=(0.20, 0.02978, 0.05000),
            rot=(0.49920, -0.49920, 0.50080, -0.50080),
            convention="world",
        ),
    )

    # rgb_head_center — SenYun SM5M12NJ (front interaction RGB, 110° HFOV).
    rgb_head_center = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/head_pitch_link/rgb_head_center_cam",
        update_period=0.05,
        height=768,
        width=1024,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=7.34, focus_distance=400.0,
            horizontal_aperture=20.955, clipping_range=(0.05, 20.0),
        ),
        offset=CameraCfg.OffsetCfg(
            # URDF-spec x=0.06840 → 0.20 (pushed +14cm forward of head shell).
            pos=(0.20, -0.00022, 0.05000),
            rot=(0.49920, -0.49920, 0.50080, -0.50080),
            convention="world",
        ),
    )

    # rgb_head_rear — SenYun SM3S23NS (rear-facing RGB, 156° HFOV).
    rgb_head_rear = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/head_pitch_link/rgb_head_rear_cam",
        update_period=0.05,
        height=768,
        width=1024,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.23, focus_distance=400.0,
            horizontal_aperture=20.955, clipping_range=(0.05, 20.0),
        ),
        offset=CameraCfg.OffsetCfg(
            # URDF-spec x=-0.08340 → -0.20 (pushed −12cm rear of head shell).
            pos=(-0.20, 0.00026, 0.00000),
            rot=(0.50080, -0.50080, -0.49920, 0.49920),
            convention="world",
        ),
    )


##
# Actions
##
@configclass
class ActionsCfg:
    """Action: Pink IK for both arms + direct hand joint positions."""

    upper_body_ik = PinkInverseKinematicsActionCfg(
        pink_controlled_joint_names=_LEFT_ARM_JOINTS + _RIGHT_ARM_JOINTS,
        hand_joint_names=_HAND_JOINTS,
        target_eef_link_names={
            "left_wrist": LEFT_EEF_LINK,
            "right_wrist": RIGHT_EEF_LINK,
        },
        asset_name="robot",
        controller=PinkIKControllerCfg(
            articulation_name="robot",
            # X2 URDF has <link name="base_link"/> as the actual root with a
            # fixed `base_link_joint` to `pelvis`. merge_fixed_joints=True bakes
            # pelvis INTO base_link, so the surviving root body is `base_link`.
            base_link_name="base_link",
            num_hand_joints=len(_HAND_JOINTS),
            show_ik_warnings=False,
            fail_on_joint_limit_violation=False,
            variable_input_tasks=[
                # Tuned to match GR1T2's working baseline:
                #   FrameTask: position_cost=8.0, orientation_cost=1.0, lm_damping=12, gain=0.5
                #   DampingTask: cost=0.5 (regularizes joint velocity, prevents twitchy wrist)
                #   NullSpacePostureTask: cost=0.5, lm_damping=1 (regularizes arm posture
                #     toward zero so the solver doesn't drift into awkward configurations).
                # IK weights copied from GR1T2 (pickplace_gr1t2_env_cfg.py):
                #   FrameTask: position_cost=8.0, orientation_cost=1.0  (8:1 in
                #     favor of position so wrist orientation tracks softly).
                #   DampingTask: cost=0.5 (joint-velocity regularization).
                #   NullSpacePostureTask: cost=0.5, lm_damping=1 — soft nudge of
                #     shoulders + elbows + waist toward init pose; lets the IK
                #     pick natural arm postures without hard-pinning the body.
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
                    # _LEFT_ARM_JOINTS[:4] = shoulder pitch/roll/yaw + elbow.
                    # Waist included to mirror GR1T2 — listing is a no-op
                    # since waist isn't in pink_controlled_joint_names, but
                    # kept for parity. Physical waist lock remains via the
                    # high-PD actuator in x2.py.
                    controlled_joints=(
                        _LEFT_ARM_JOINTS[:4] + _RIGHT_ARM_JOINTS[:4] + _WAIST_JOINTS
                    ),
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

        hand_joint_state = ObsTerm(
            func=mdp.get_robot_joint_state, params={"joint_names": ["R_.*", "L_.*"]}
        )
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
        stereo_head_front_rgb = ObsTerm(
            func=base_mdp.image,
            params={"sensor_cfg": SceneEntityCfg("stereo_head_front"), "data_type": "rgb", "normalize": False},
        )
        rgb_head_center_rgb = ObsTerm(
            func=base_mdp.image,
            params={"sensor_cfg": SceneEntityCfg("rgb_head_center"), "data_type": "rgb", "normalize": False},
        )
        rgb_head_rear_rgb = ObsTerm(
            func=base_mdp.image,
            params={"sensor_cfg": SceneEntityCfg("rgb_head_rear"), "data_type": "rgb", "normalize": False},
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
        params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("object")},
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
            "grasp_joint_names": ("R_thumb_mcp_joint", "R_index_pip_joint"),
            "grasp_closed_threshold": 0.3,
            "grasp_proximity": 0.15,
            "hand_body_name": RIGHT_EEF_LINK,
        },
    )


##
# Events
##
@configclass
class EventCfg:
    # Runs once at env creation — hides the plastic crate / packaged props that
    # ship inside the PackingTable USD so the table surface is clean.
    hide_crate = EventTerm(func=_hide_packing_table_crate, mode="startup")

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # Place can 22cm to the RIGHT of env origin (≈ tray center) with ±5cm
    # x/y jitter, so the can shows up around (0.22, 0.40) ± 5cm each axis.
    # Tray init_state is the fixed pose; no tray reset event = tray stays put.
    reset_object = EventTerm(
        func=_reset_object_left_or_right_of_tray,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("object"), "offset_m": 0.22, "jitter_m": 0.05},
    )


##
# Env cfg
##
@configclass
class PickPlaceX2EnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for X2 (with OmniHand) place-can-into-tray environment."""

    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events = EventCfg()

    commands = None
    rewards = None
    curriculum = None

    xr: XrCfg = XrCfg(anchor_pos=(0.0, 0.0, 0.0), anchor_rot=(1.0, 0.0, 0.0, 0.0))

    NUM_OPENXR_HAND_JOINTS = 26
    temp_urdf_dir = tempfile.gettempdir()

    # OmniHand grasp proxy: thumb_mcp + index_pip as 2-joint stand-in for the
    # parallel-gripper check in place_after_grasp / object_a_is_into_b.
    gripper_joint_names = ["R_thumb_mcp_joint", "R_index_pip_joint"]
    gripper_open_val = 0.0
    gripper_threshold = 0.5

    # Idle action: left wrist pose (7) + right wrist pose (7) + 32 hand joints (0 = open).
    idle_action = torch.tensor(
        [
            0.22, 0.30, 1.10, 1.0, 0.0, 0.0, 0.0,
            -0.22, 0.30, 1.10, 1.0, 0.0, 0.0, 0.0,
            *([0.0] * 32),
        ]
    )

    def __post_init__(self):
        self.decimation = 6
        self.episode_length_s = 20.0
        self.sim.dt = 1 / 120
        self.sim.render_interval = 2  # match GR1T2 — halves render cost, freeing GPU for tracking

        # Use the merged URDF (X2 body + OmniHand T2) directly for Pink IK.
        self.actions.upper_body_ik.controller.urdf_path = (
            "/home/wagner/code/IsaacLab/assets/X2_omnihand/X2_omnihand.urdf"
        )
        # Mesh root: parent of the meshes/ dir — matches the relative paths inside
        # the merged URDF (meshes/x2/... for body, meshes/omnihand/... for hands).
        self.actions.upper_body_ik.controller.mesh_path = (
            "/home/wagner/code/IsaacLab/assets/X2_omnihand"
        )

        # Wrist compensation parameters used by the retargeter at runtime.
        # NOTE: these only affect IK when real hand tracking flows. Pre-teleop
        # robot pose comes from `joint_pos` in InitialStateCfg above.
        # L: identity — URDF palm_joint rpy=(0, π, +π/2) handles L mounting.
        # R: +π Z to compensate for URDF palm_joint rpy=(0, π, -π/2). The yaw
        # delta between L and R URDFs is π, so R's IK target needs to be
        # rotated 180° around Z so the rendered OmniHand mirrors L's gesture.
        left_fixed_rpy = (0.0, 0.0, 0.0)
        right_fixed_rpy = (0.0, 0.0, 3.1415927)
        left_shoulder_pos = (-0.2, 0.05, 1.4)
        right_shoulder_pos = (0.2, 0.05, 1.4)
        max_reach = 0.7

        # Idle wrist poses (still used by keyboard device's left_wrist_idle /
        # right_wrist_start). Compute from idle_action so it stays consistent.
        idle = self.idle_action.tolist()
        left_idle_pose = tuple(idle[0:7])
        right_idle_pose = tuple(idle[7:14])

        self.teleop_devices = DevicesCfg(
            devices={
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        A2OmniHandRetargeterCfg(
                            enable_visualization=True,
                            num_open_xr_hand_joints=2 * self.NUM_OPENXR_HAND_JOINTS,
                            sim_device=self.sim.device,
                            hand_joint_names=_HAND_JOINTS,
                            left_fixed_rpy=left_fixed_rpy,
                            right_fixed_rpy=right_fixed_rpy,
                            left_shoulder_pos=left_shoulder_pos,
                            right_shoulder_pos=right_shoulder_pos,
                            max_reach=max_reach,
                            debug_every=30,
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
                "keyboard": A2OmniHandKeyboardCfg(
                    sim_device=self.sim.device,
                    pos_sensitivity=0.4,
                    rot_sensitivity=0.8,
                    left_wrist_idle=left_idle_pose,
                    right_wrist_start=right_idle_pose,
                    dt=1.0 / 30.0,
                ),
            }
        )
