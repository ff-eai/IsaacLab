# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Env for A2 humanoid with OmniHand T2 hands: place can into tray via VR teleop.

Forked from pickplace_a2_env_cfg.py. Uses A2_OMNIHAND_NOMIMIC_CFG and the
OmniHand dex-retargeting path (DexPilot, 10 DoF per hand driven; pip/dip joints
populated via mimic ratios from the URDF in the retargeter output).

Known tuning points (will fail/misbehave until validated on your rig):
  * Pink IK frame names: default prefix is "a2_t2d0_ultra_" derived from the URDF
    <robot name="..."> tag. If Pink IK raises a KeyError for the frame name on
    first run, print the available frames (`controller.robot.frames`) and update
    PINK_IK_FRAME_PREFIX below.
  * Wrist (right-hand) 180° z-flip is copied from GR1T2; may need retuning for A2
    hand URDF orientation — check by inspecting hand gizmo on first teleop session.
  * Can / tray USD paths are Nucleus defaults; override if Nucleus has different
    YCB / KitchenSet layouts.
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

from isaaclab_assets.robots.a2 import A2_OMNIHAND_NOMIMIC_CFG  # isort: skip


def _fwd_left_to_quat_wxyz(forward, left):
    """Build an (w, x, y, z) quaternion from RoboTwin's (forward, left) basis.
    Uses Isaac Lab CameraCfg 'world' convention (X=forward, Y=left, Z=up) so the
    RoboTwin config ports 1:1. Normalizes inputs, orthogonalizes, and returns
    the quat for the rotation whose X/Y/Z columns are forward/left/up."""
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    fwd = np.asarray(forward, dtype=np.float64)
    fwd = fwd / np.linalg.norm(fwd)
    lft = np.asarray(left, dtype=np.float64)
    lft = lft / np.linalg.norm(lft)
    up = np.cross(fwd, lft)
    up = up / np.linalg.norm(up)
    # Re-orthogonalize left so basis is perfectly orthonormal
    lft = np.cross(up, fwd)
    rot_mat = np.column_stack([fwd, lft, up])
    qxyzw = R.from_matrix(rot_mat).as_quat()
    return (float(qxyzw[3]), float(qxyzw[0]), float(qxyzw[1]), float(qxyzw[2]))


def _reset_object_random_side(
    env,
    env_ids,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    tray_anchor_xy: tuple[float, float] = (0.0, 0.35),
    spawn_z: float = 1.00,
    x_offset: float = 0.20,
    force_side: str | None = None,
):
    """Place the bottle 20 cm to the LEFT or RIGHT of the tray.

    ``force_side``: ``"left"``, ``"right"``, or ``None`` for 50/50 random per env.
    Tray anchor xy is hard-coded (matches tray's init_state pos). Spawning slightly
    above the table top so gravity settles the bottle. Quaternion matches the
    upright orientation used by the bottle's init_state.
    """
    import torch as _torch
    asset = env.scene[asset_cfg.name]
    n = len(env_ids)

    if force_side == "right":
        sign = _torch.ones((n,), device=env.device, dtype=_torch.float32)
    elif force_side == "left":
        sign = -_torch.ones((n,), device=env.device, dtype=_torch.float32)
    else:
        # Coin flip per env: 0 → left (-x), 1 → right (+x)
        side = _torch.randint(0, 2, (n,), device=env.device, dtype=_torch.int64)
        sign = (2 * side.to(_torch.float32) - 1.0)  # -1 (left) or +1 (right)

    pos = _torch.zeros((n, 3), device=env.device, dtype=_torch.float32)
    pos[:, 0] = tray_anchor_xy[0] + sign * x_offset
    pos[:, 1] = tray_anchor_xy[1]
    pos[:, 2] = spawn_z

    # Add the env-origin offset (envs are tiled; world pose = origin + local).
    env_origins = env.scene.env_origins[env_ids]
    pos = pos + env_origins

    # Upright orientation, same as init_state: quat (w,x,y,z) = (0.707, 0.707, 0, 0).
    quat = _torch.tensor([0.7071, 0.7071, 0.0, 0.0], device=env.device, dtype=_torch.float32)
    quat = quat.unsqueeze(0).repeat(n, 1)

    pose = _torch.cat([pos, quat], dim=-1)
    asset.write_root_pose_to_sim(pose, env_ids=env_ids)
    asset.write_root_velocity_to_sim(
        _torch.zeros((n, 6), device=env.device, dtype=_torch.float32),
        env_ids=env_ids,
    )


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


# --- Tunables ---------------------------------------------------------------
# Frame names passed to Pink IK are resolved directly against pinocchio's parsed
# URDF — no articulation-name prefix is prepended (verified with pink_diag script:
# "left_arm_link07" → frame ID 46, "right_arm_link07" → frame ID 90).
PINK_IK_FRAME_PREFIX = ""
LEFT_EEF_LINK = "left_arm_link07"   # URDF left_hand merged into arm_link07 by merge_fixed_joints
RIGHT_EEF_LINK = "right_arm_link07"

# RoboTwin 001_bottle/base13 — Coca-Cola bottle, converted from
# /home/wagner/code/RoboTwin/assets/objects/001_bottle/visual/base13.glb via
# scripts/demos/convert_robotwin_assets.py. Standing upright with the rot quat
# below the GLB y-axis (long axis) maps to world-z. Bottle is ~25 cm tall at
# scale 0.132; GLB origin sits at the bottle's bottom.
CAN_USD = "/home/wagner/code/IsaacLab/assets/robotwin/001_bottle_base13.usd"
# RoboTwin 008_tray (matches place_can_basket task).
TRAY_USD = "/home/wagner/code/IsaacLab/assets/robotwin/008_tray_base0.usd"

# Per-side arm joint names — A2 URDF uses the idxNN_ prefix from the SolidWorks exporter.
_LEFT_ARM_JOINTS = [f"idx{13 + i:02d}_left_arm_joint{i + 1}" for i in range(7)]
_RIGHT_ARM_JOINTS = [f"idx{20 + i:02d}_right_arm_joint{i + 1}" for i in range(7)]

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

    # Procedural table — 0.7 × 0.5 × 0.05 m (length × width × thickness).
    # Tabletop top at z=0.90 (+20 cm from original 0.70).
    packing_table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable",
        # tabletop center = top_z - thickness/2 = 0.90 - 0.025 = 0.875
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.45, 0.875), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.7, 0.5, 0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.95, 0.95), roughness=0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    # 4 table legs, 0.1 × 0.1 × 0.85 each (floor z=0 → tabletop bottom z=0.85).
    # Leg positions: x = ±0.30, y = ±0.20 around table center y=0.45.
    table_leg_0 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/TableLeg0",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.30, 0.45 + 0.20, 0.425)),
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.1, 0.1, 0.85),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.95, 0.95), roughness=0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )
    table_leg_1 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/TableLeg1",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.30, 0.45 - 0.20, 0.425)),
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.1, 0.1, 0.85),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.95, 0.95), roughness=0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )
    table_leg_2 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/TableLeg2",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.30, 0.45 + 0.20, 0.425)),
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.1, 0.1, 0.85),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.95, 0.95), roughness=0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )
    table_leg_3 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/TableLeg3",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.30, 0.45 - 0.20, 0.425)),
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.1, 0.1, 0.85),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.95, 0.95), roughness=0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    # RoboTwin 001_bottle/base13 — Coca-Cola bottle. Quaternion [0.707, 0.707, 0, 0]
    # is the same upright orientation used for the can swap. Bottle bottom spawns
    # ~10 cm above the table and gravity settles it onto the surface. Mass set to
    # 0.30 kg (closer to a real-world ~0.5 L Coke bottle) so the hand has to grip
    # rather than flick it. Position (0.21, 0.43) — moved 20 cm along the
    # straight line toward the tray (0.0, 0.35) from the previous (0.40, 0.50).
    # Bottle-to-tray distance dropped 0.43 m → 0.23 m.
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.21, 0.33, 1.00], rot=[0.707, 0.707, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=CAN_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.30),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )

    # RoboTwin 008_tray — same asset used in place_can_basket. Mass 0.85 kg per
    # place_can_basket.py. Kinematic_enabled=True keeps it stable on the table
    # while the can is placed into it (avoids PhysX velocity-on-kinematic warning
    # from reset events as the 'hide_crate' isn't running on this asset).
    tray = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Tray",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.0, 0.25, 0.90], rot=[0.707, 0.707, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=TRAY_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.85),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )

    robot: ArticulationCfg = A2_OMNIHAND_NOMIMIC_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0.05, 0.93),  # fixed pelvis height; table raised +50 cm below arms
            rot=(0.7071, 0, 0, 0.7071),  # face +y (towards table)
            joint_pos={
                # Arms: rest pose copied from the real-robot training dataset
                # (zhiyuandata/data/chunk-000/episode_000000.parquet, frame 0,
                # state[314:328]). Matches the policy's training distribution
                # so state.arm at t=0 is in-distribution. Notes:
                #   joint2 mirrored ±1.20 — main "fold" via shoulder pitch
                #   joint4 mirrored ±0.10 — slight elbow bend
                #   joint5 +0.90 BOTH sides (NOT mirrored) — wrist roll
                "idx13_left_arm_joint1":   0.0002,
                "idx14_left_arm_joint2":   1.1995,
                "idx15_left_arm_joint3":  -0.0002,
                "idx16_left_arm_joint4":  -0.0994,
                "idx17_left_arm_joint5":   0.8993,
                "idx18_left_arm_joint6":  -0.0005,
                "idx19_left_arm_joint7":  -0.0003,
                "idx20_right_arm_joint1": -0.0002,
                "idx21_right_arm_joint2": -1.2003,
                "idx22_right_arm_joint3":  0.0002,
                "idx23_right_arm_joint4":  0.1001,
                "idx24_right_arm_joint5":  0.9005,
                "idx25_right_arm_joint6": -0.0006,
                "idx26_right_arm_joint7": -0.0010,
                # Legs: explicitly straight. A2.urdf's all-zero rest pose has a
                # built-in bend on the left leg; force the chain straight by
                # pinning each joint to 0 (gravity disabled on the robot so
                # legs stay wherever we put them).
                "idx01_left_hip_roll": 0.0,
                "idx02_left_hip_yaw": 0.0,
                "idx03_left_hip_pitch": 0.0,
                "idx04_left_tarsus": 0.0,
                "idx05_left_toe_pitch": 0.0,
                "idx06_left_toe_roll": 0.0,
                "idx07_right_hip_roll": 0.0,
                "idx08_right_hip_yaw": 0.0,
                "idx09_right_hip_pitch": 0.0,
                "idx10_right_tarsus": 0.0,
                "idx11_right_toe_pitch": 0.0,
                "idx12_right_toe_roll": 0.0,
                # Head neutral.
                "idx2[78]_head_joint.*": 0.0,
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

    # --- Cameras (A2 external, per RoboTwin /envs/camera spec) -----------------
    # Head camera, 43° FOV, 1280×720. NOTE: originally mounted on head_link02 but
    # that link's transform is NaN in this OmniHand USD build (snapshot showed
    # the camera renders pure black), so we mount on base_link instead and
    # offset z to head height (~0.50 m above pelvis). Aimed at the table center
    # (world ≈ 0, 0.40, 0.70). robot's local +x = world +y (forward).
    head_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/head_camera",
        update_period=0.05,
        height=720,
        width=1280,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0, focus_distance=400.0,
            horizontal_aperture=20.955, clipping_range=(0.05, 20.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.14, 0.0, 0.50),
            rot=_fwd_left_to_quat_wxyz(forward=[0.42, 0.0, -0.91], left=[0.0, 1.0, 0.0]),
            convention="world",
        ),
    )

    # Chest left camera: wide 120° fisheye, 640×480, RGB only, mounted on base_link.
    chest_left_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/chest_left_camera",
        update_period=0.05,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=6.0, focus_distance=400.0,
            horizontal_aperture=20.955, clipping_range=(0.05, 20.0),  # short focal for wide FOV
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.1, 0.02, 0.4),
            rot=_fwd_left_to_quat_wxyz(forward=[0.6941, -0.0252, -0.7194], left=[0.0363, 0.9993, 0.0]),
            convention="world",
        ),
    )

    chest_right_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/chest_right_camera",
        update_period=0.05,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=6.0, focus_distance=400.0,
            horizontal_aperture=20.955, clipping_range=(0.05, 20.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.1, -0.08, 0.4),
            rot=_fwd_left_to_quat_wxyz(forward=[0.6941, -0.0252, -0.7194], left=[0.0363, 0.9993, 0.0]),
            convention="world",
        ),
    )


def enable_pickplace_omnihand_5cameras(cfg: "PickPlaceA2OmniHandEnvCfg") -> None:
    """Add wrist RGB cameras for GR00T v5 (hand_left / hand_right). Head+chest already on scene."""
    # Wrist camera pose + lens synced from ~/Documents/wrist_camera.usd (Isaac Sim stage
    # export). The pos/rot below are the camera prims' local transforms relative to
    # *_arm_link07, expressed in the OpenGL / Usd.Camera convention (forward -Z, up +Y),
    # so they are applied verbatim with convention="opengl". Poses are asymmetric L/R as
    # tuned in the GUI; lens (focal_length / horizontal_aperture / clipping_range) is also
    # taken from the USD (≈89° HFOV). Resolution stays 640×480 (GR00T hand_left/right).
    scene = cfg.scene
    scene.left_wrist_camera = CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{LEFT_EEF_LINK}/left_wrist_camera",
        update_period=0.05,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=0.5103046298027039, focus_distance=400.0,
            horizontal_aperture=1.0, clipping_range=(0.02, 3.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.06, 0.14, 0.0),
            rot=(0.8323982695412339, -0.2899999063978923, 0.4716783164260747, -0.023081182106573617),
            convention="opengl",
        ),
    )
    scene.right_wrist_camera = CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{RIGHT_EEF_LINK}/right_wrist_camera",
        update_period=0.05,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=0.5111332535743713, focus_distance=400.0,
            horizontal_aperture=1.0, clipping_range=(0.02, 3.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(-0.04, 0.15, -0.04),
            rot=(0.6987029315587729, -0.2622516746261983, -0.6254976559562064, -0.22757626189965985),
            convention="opengl",
        ),
    )
    pol = cfg.observations.policy
    pol.left_wrist_camera_rgb = ObsTerm(
        func=base_mdp.image,
        params={"sensor_cfg": SceneEntityCfg("left_wrist_camera"), "data_type": "rgb", "normalize": False},
    )
    pol.right_wrist_camera_rgb = ObsTerm(
        func=base_mdp.image,
        params={"sensor_cfg": SceneEntityCfg("right_wrist_camera"), "data_type": "rgb", "normalize": False},
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
                    # Regularize SHOULDER + ELBOW joints only (joints 1-4 per arm).
                    # Wrist joints 5/6/7 must remain free so orientation tracking can
                    # use both wrist DoFs. Including them caused only one wrist joint
                    # Shoulders only (joints 1-3) — elbow is *not* regularized
                    # so the IK can extend it freely toward 0 (straight) when
                    # the wrist target is far forward.
                    controlled_joints=_LEFT_ARM_JOINTS[:3] + _RIGHT_ARM_JOINTS[:3],
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

        head_camera_rgb = ObsTerm(
            func=base_mdp.image,
            params={"sensor_cfg": SceneEntityCfg("head_camera"), "data_type": "rgb", "normalize": False},
        )
        head_camera_depth = ObsTerm(
            func=base_mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("head_camera"),
                "data_type": "distance_to_image_plane",
                "normalize": False,
            },
        )
        chest_left_camera_rgb = ObsTerm(
            func=base_mdp.image,
            params={"sensor_cfg": SceneEntityCfg("chest_left_camera"), "data_type": "rgb", "normalize": False},
        )
        chest_right_camera_rgb = ObsTerm(
            func=base_mdp.image,
            params={"sensor_cfg": SceneEntityCfg("chest_right_camera"), "data_type": "rgb", "normalize": False},
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
            # Accept a left-hand grasp-then-place too: the trained policy on
            # 18003 reaches with the LEFT arm despite the can being on the
            # right side, so a right-hand-only success criterion fires never.
            "grasp_joint_names_alt": ("L_thumb_mcp_joint", "L_index_pip_joint"),
            "hand_body_name_alt": LEFT_EEF_LINK,
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

    # Bottle reset: pick LEFT or RIGHT side of the tray (50/50 per env_reset)
    # at exactly x_offset from the tray, so the policy faces a clear bimodal
    # task that exercises whichever arm is closer.
    reset_object = EventTerm(
        func=_reset_object_random_side,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "tray_anchor_xy": (0.0, 0.25),
            "spawn_z": 1.00,
            "x_offset": 0.20,
        },
    )

    # Tray reset: deterministic — keep at init_state pos so the bottle's "20 cm
    # from tray" geometry holds.
    reset_tray = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": [0.0, 0.0], "y": [0.0, 0.0]},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("tray"),
        },
    )


##
# Env cfg
##
@configclass
class PickPlaceA2OmniHandEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for A2 place-can-into-tray environment."""

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

        # Use the merged nomimic URDF (A2 body + OmniHand T2) directly for Pink IK.
        self.actions.upper_body_ik.controller.urdf_path = (
            "/home/wagner/code/IsaacLab/assets/A2_omnihand/A2_omnihand_nomimic.urdf"
        )
        # Mesh root: parent of the meshes/ dir — matches the relative paths inside
        # the merged URDF (meshes/public/... for body, meshes/omnihand/... for hands).
        self.actions.upper_body_ik.controller.mesh_path = (
            "/home/wagner/code/IsaacLab/assets/A2_omnihand"
        )

        # Pull idle wrist poses straight from idle_action so the keyboard device
        # starts at a pose Pink IK already considers feasible.
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
                            # Aligned with X2's OmniHand mounting (URDF palm rpy now
                            # matches X2: (0, π, ±π/2)). Both sides set yaw=+π/2
                            # to compensate for residual wrist rotation observed
                            # after the URDF palm-mount alignment.
                            left_fixed_rpy=(0.0, 0.0, 1.5707963),
                            right_fixed_rpy=(0.0, 0.0, 1.5707963),
                            left_shoulder_pos=(-0.2, 0.05, 1.1),
                            right_shoulder_pos=(0.2, 0.05, 1.1),
                            # Stretch reach: pos_scale multiplies the user→robot
                            # offset from the shoulder; max_reach widens the
                            # clip sphere so scaled offsets aren't truncated.
                            pos_scale=1.3,
                            max_reach=0.9,
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
