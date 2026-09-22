# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Env for A2 humanoid with stock s6_hand: place can into tray via VR teleop.

Cloned from pickplace_a2_omnihand_env_cfg.py with the same Pink IK tuning,
but driving A2_NOMIMIC_CFG (s6_hand) instead of OmniHand T2. Uses the s6
A2 retargeter path (A2DexRetargeting; mimic coupling reimposed inside the
retargeter output).

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
from isaaclab.devices.openxr.retargeters.humanoid.agibot.a2_keyboard_teleop import A2KeyboardCfg
from isaaclab.devices.openxr.retargeters.humanoid.agibot.a2_retargeter import A2RetargeterCfg
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

from isaaclab_assets.robots.a2 import A2_NOMIMIC_CFG  # isort: skip


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


def enable_pickplace_a2_cameras(cfg: "PickPlaceA2EnvCfg") -> None:
    """Turn on head + chest RGB cameras (same mounts as pickplace_a2_omnihand_env_cfg)."""
    scene = cfg.scene
    scene.head_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/head_camera",
        update_period=0.05,
        height=720,
        width=1280,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 20.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.14, 0.0, 0.50),
            rot=_fwd_left_to_quat_wxyz(forward=[0.42, 0.0, -0.91], left=[0.0, 1.0, 0.0]),
            convention="world",
        ),
    )
    scene.chest_left_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/chest_left_camera",
        update_period=0.05,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=6.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 20.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.1, 0.02, 0.4),
            rot=_fwd_left_to_quat_wxyz(forward=[0.6941, -0.0252, -0.7194], left=[0.0363, 0.9993, 0.0]),
            convention="world",
        ),
    )
    scene.chest_right_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/chest_right_camera",
        update_period=0.05,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=6.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 20.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.1, -0.08, 0.4),
            rot=_fwd_left_to_quat_wxyz(forward=[0.6941, -0.0252, -0.7194], left=[0.0363, 0.9993, 0.0]),
            convention="world",
        ),
    )

    pol = cfg.observations.policy
    pol.head_camera_rgb = ObsTerm(
        func=base_mdp.image,
        params={"sensor_cfg": SceneEntityCfg("head_camera"), "data_type": "rgb", "normalize": False},
    )
    pol.head_camera_depth = ObsTerm(
        func=base_mdp.image,
        params={
            "sensor_cfg": SceneEntityCfg("head_camera"),
            "data_type": "distance_to_image_plane",
            "normalize": False,
        },
    )
    pol.chest_left_camera_rgb = ObsTerm(
        func=base_mdp.image,
        params={"sensor_cfg": SceneEntityCfg("chest_left_camera"), "data_type": "rgb", "normalize": False},
    )
    pol.chest_right_camera_rgb = ObsTerm(
        func=base_mdp.image,
        params={"sensor_cfg": SceneEntityCfg("chest_right_camera"), "data_type": "rgb", "normalize": False},
    )


def enable_pickplace_a2_5cameras(cfg: "PickPlaceA2EnvCfg") -> None:
    """Head + chest + wrist RGB (GR00T v5 5cam_abs_ee training layout)."""
    enable_pickplace_a2_cameras(cfg)
    _wrist_fwd = [0.0, -0.574, 0.819]  # link +Z pitched ~35° toward fingers (-Y)
    scene = cfg.scene
    scene.left_wrist_camera = CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{LEFT_EEF_LINK}/left_wrist_camera",
        update_period=0.05,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=6.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 20.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.05),
            rot=_fwd_left_to_quat_wxyz(forward=_wrist_fwd, left=[0.0, 1.0, 0.0]),
            convention="world",
        ),
    )
    scene.right_wrist_camera = CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{RIGHT_EEF_LINK}/right_wrist_camera",
        update_period=0.05,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=6.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 20.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.05),
            rot=_fwd_left_to_quat_wxyz(forward=_wrist_fwd, left=[0.0, -1.0, 0.0]),
            convention="world",
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


def _compute_tray_top_z(env, env_ids):
    """Startup event — query the tray prim's world AABB max.z and stash it as
    `env._tray_top_z` (per env_id, but trays don't move so a single scalar
    suffices in practice — we store a tensor of shape (num_envs,)).

    Wrapped in try/except so a USD-level failure here can't block scene
    creation; falls back to a conservative tray_root + 5 cm estimate."""
    fallback = 1.05  # tray root z=1.00 + ~5 cm rim
    tops = [fallback] * env.num_envs
    try:
        import omni.usd
        from pxr import UsdGeom

        stage = omni.usd.get_context().get_stage()
        bbox_cache = UsdGeom.BBoxCache(0, ["default", "render"], useExtentsHint=False)
        for env_i in range(env.num_envs):
            try:
                prim = stage.GetPrimAtPath(f"/World/envs/env_{env_i}/Tray")
                if prim and prim.IsValid():
                    world_bbox = bbox_cache.ComputeWorldBound(prim)
                    tops[env_i] = float(world_bbox.GetRange().GetMax()[2])
            except Exception as exc:
                print(f"[tray_top_z] env {env_i} failed: {exc!r} — using {fallback}")
    except Exception as exc:
        print(f"[tray_top_z] global failure: {exc!r} — using fallback {fallback} for all envs")
    env._tray_top_z = torch.tensor(tops, device=env.device, dtype=torch.float32)
    print(f"[tray_top_z] {tops}  (stashed on env._tray_top_z)")


# Color palette for can + tray randomization (R, G, B in [0, 1]).
_COLOR_PALETTE = [
    (0.85, 0.10, 0.10),   # red
    (0.10, 0.40, 0.85),   # blue
    (0.10, 0.70, 0.20),   # green
    (0.95, 0.85, 0.10),   # yellow
    (0.65, 0.20, 0.85),   # purple
    (0.95, 0.55, 0.10),   # orange
    (0.10, 0.75, 0.75),   # teal
    (0.85, 0.10, 0.55),   # magenta
]


def _set_prim_diffuse_recursive(prim, color, _UsdShade, _Gf):
    """Walk a prim subtree, find every Material's PBR/PreviewSurface shader, and
    override the diffuse-color input with the given (r, g, b). Each per-prim
    failure is swallowed (logged once) so partial success is acceptable."""
    stack = [prim]
    while stack:
        cur = stack.pop()
        try:
            if not cur or not cur.IsValid():
                continue
            for child in cur.GetAllChildren():
                stack.append(child)
            if cur.GetTypeName() != "Material":
                continue
            mat = _UsdShade.Material(cur)
            shader = mat.ComputeSurfaceSource()[0]
            if not shader:
                for s in cur.GetChildren():
                    if s.GetTypeName() == "Shader":
                        shader = _UsdShade.Shader(s)
                        break
            if not shader:
                continue
            for input_name in (
                "diffuseColor",
                "diffuse_color_constant",
                "base_color",
            ):
                inp = shader.GetInput(input_name)
                if inp:
                    try:
                        inp.Set(_Gf.Vec3f(*color))
                    except Exception:
                        pass
        except Exception as exc:
            # log only once per call so we don't spam
            if not getattr(_set_prim_diffuse_recursive, "_warned", False):
                print(f"[colorize] prim walk error: {exc!r} (further errors suppressed)")
                _set_prim_diffuse_recursive._warned = True


def _randomize_can_tray_colors(env, env_ids):
    """Reset event — pick a random color from `_COLOR_PALETTE` per env and
    override diffuse-color on every material under the can and tray prims.
    Uniform sampling, separate draws for can and tray. Wrapped in try/except
    so a USD-level failure can't block the env reset."""
    try:
        import omni.usd
        from pxr import UsdShade, Gf
    except Exception as exc:
        print(f"[colorize] import failed: {exc!r}")
        return

    try:
        stage = omni.usd.get_context().get_stage()
    except Exception as exc:
        print(f"[colorize] stage fetch failed: {exc!r}")
        return

    env_indices = range(env.num_envs) if env_ids is None else env_ids.cpu().tolist()
    for env_i in env_indices:
        for asset_name in ("Object", "Tray"):
            try:
                prim = stage.GetPrimAtPath(f"/World/envs/env_{env_i}/{asset_name}")
                if not prim or not prim.IsValid():
                    continue
                color = _COLOR_PALETTE[
                    int(torch.randint(0, len(_COLOR_PALETTE), (1,)).item())
                ]
                _set_prim_diffuse_recursive(prim, color, UsdShade, Gf)
            except Exception as exc:
                print(f"[colorize] env {env_i} {asset_name} failed: {exc!r}")


# --- Tunables ---------------------------------------------------------------
# Frame names passed to Pink IK are resolved directly against pinocchio's parsed
# URDF — no articulation-name prefix is prepended (verified with pink_diag script:
# "left_arm_link07" → frame ID 46, "right_arm_link07" → frame ID 90).
PINK_IK_FRAME_PREFIX = ""
LEFT_EEF_LINK = "left_arm_link07"   # URDF left_hand merged into arm_link07 by merge_fixed_joints
RIGHT_EEF_LINK = "right_arm_link07"

# RoboTwin 071_can (matches place_can_basket task) — converted from
# /home/wagner/code/RoboTwin/assets/objects/071_can/visual/base0.glb via
# scripts/demos/convert_robotwin_assets.py.
CAN_USD = "/home/wagner/code/IsaacLab/assets/robotwin/071_can_base0.usd"
# RoboTwin 008_tray (matches place_can_basket task).
TRAY_USD = "/home/wagner/code/IsaacLab/assets/robotwin/008_tray_base0.usd"

# Per-side arm joint names — A2 URDF uses the idxNN_ prefix from the SolidWorks exporter.
_LEFT_ARM_JOINTS = [f"idx{13 + i:02d}_left_arm_joint{i + 1}" for i in range(7)]
_RIGHT_ARM_JOINTS = [f"idx{20 + i:02d}_right_arm_joint{i + 1}" for i in range(7)]

# Virtual waist (yaw) DOF inserted between base_link and the arm bases in the
# URDF — gives Pink IK an extra reach axis. Matches GR1T2's waist_yaw_joint role.
_WAIST_JOINTS = ["waist_yaw_joint"]

# Hand joints in the EXACT order the A2 articulation exposes them. This matters
# because Isaac Lab's PinkInverseKinematicsAction calls find_joints(preserve_order=False),
# which returns joint ids in the asset's internal ordering. If hand_joint_names
# is in any other order, retargeter output gets written to the wrong joints and
# fingers don't follow the hand tracker.
#
# 12 actuator joints (6 per hand): thumb_swing + thumb_1 + (index/middle/ring/pinky)_1.
# Matches AimRT firmware: thumb_swing=*_thumb_0, thumb_1=*_thumb_1, *_1=*_<finger>.
# The mimic *_2 / *_3 / per-finger *_2 joints are converted to FixedJoint in
# A2_nomimic_physics.usd so they don't appear in the articulation DOF list.
# Reduced 6-DOF hand (3 per side): thumb_swing + thumb_1 + index_1.
# index_1 acts as the "grip" proxy — the middle/ring/pinky finger joints
# (L_middle_1, L_ring_1, L_pinky_1 and R analogues) are NOT in the action
# term and stay at their init value (0 = open). For a true 4-finger grip,
# add a runtime event/action term that mirrors L_index_1 → other fingers.
#
# IMPORTANT: order MUST match the asset's internal joint order, because
# PinkInverseKinematicsAction calls find_joints(preserve_order=False) which
# returns joint ids sorted by asset order. The retargeter writes
# action[i] → joint_targets[ joint_ids[i] ]; if our order differs from the
# asset's, every finger value lands on the wrong joint (e.g. thumb_swing
# command writes to L_index_1's target, leaving the actual thumb frozen).
# The asset's L_/R_ hand-joint order (from the original 12-joint list) is:
#   L_index_1, L_middle_1, L_pinky_1, L_ring_1, L_thumb_swing,
#   R_index_1, R_middle_1, R_pinky_1, R_ring_1, R_thumb_swing,
#   L_thumb_1, R_thumb_1
# For our 6-joint subset that's:
_HAND_JOINTS = [
    # Left s6_hand (6 actuated joints)
    "L_index_1_joint", "L_thumb_swing_joint", "L_thumb_1_joint",
    "L_middle_1_joint", "L_ring_1_joint", "L_pinky_1_joint",
    # Right s6_hand (6 actuated joints)
    "R_index_1_joint", "R_thumb_swing_joint", "R_thumb_1_joint",
    "R_middle_1_joint", "R_ring_1_joint", "R_pinky_1_joint",
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
    # Default pos parks the can to the right of the tray (x=0.30) at table-top
    # height. Reset event below picks the actual side per-episode and overrides
    # x so the can never spawns inside the tray xy footprint.
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.30, 0.40, 1.10], rot=[0.707, 0.707, 0, 0]),
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
    # Tray fixed at y=0.40 (35cm forward of robot at y=0.05). No reset
    # randomization — user requires the tray location to stay constant.
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

    robot: ArticulationCfg = A2_NOMIMIC_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0.05, 0.93),  # 5cm away from table (was +0.10)
            rot=(0.7071, 0, 0, 0.7071),  # face +y (towards table)
            joint_pos={
                # Arms: shoulder joints 1+2 folded so the arm hangs down toward
                # the ground; elbow (joint4) and wrist joints stay at 0 so the
                # arm is straight. joint2's URDF limit is asymmetric/mirrored
                # ([-0.524, 1.658] left vs [-1.658, 0.524] right), so the L/R
                # values are sign-flipped to fold each side in the same body
                # direction. Adjust the magnitudes below if the visual pose
                # isn't right; the elbow zero is at the limit edge so the arm
                # is guaranteed extended.
                # Calibrated teleop pose copied from /home/wagner/arm.txt.
                # joint4 (elbow) values are clamped to 0 — the raw file shows
                # ±16 µrad noise that's outside the one-sided URDF limits
                # (left [-2.094, 0], right [0, 2.094]).
                "idx13_left_arm_joint1":  -6.514795912313667e-05,
                "idx14_left_arm_joint2":   1.3168287075965412,
                "idx15_left_arm_joint3":   2.93092307634336e-06,
                "idx16_left_arm_joint4":   0.0,
                "idx17_left_arm_joint5":   1.5708031401524603,
                "idx18_left_arm_joint6":  -0.010082927355869357,
                "idx19_left_arm_joint7":   0.0021316442730244804,
                "idx20_right_arm_joint1": -6.516767056434484e-05,
                "idx21_right_arm_joint2": -1.3168289015235457,
                "idx22_right_arm_joint3":  2.9358321791896967e-06,
                "idx23_right_arm_joint4":  0.0,
                "idx24_right_arm_joint5":  1.570803145076061,
                "idx25_right_arm_joint6":  0.00981977532941597,
                "idx26_right_arm_joint7": -0.0021335183335535702,
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

    # --- Cameras — disabled by default (set to None to avoid RTX rendering in headless) ---
    # Head camera mounted on head_link02, 43° FOV, 1280×720, narrow front-view.
    # Tilted ~65° below horizontal so table edge is in FOV.
    # Cloned from pickplace_a2_omnihand_env_cfg.py (line 358): chest-mounted
    # wide-FOV downward-looking camera on base_link instead of the prior
    # head_link02-mounted narrow camera. focal_length=12mm → ~82° HFOV at
    # 1280×720 (vs prior 24mm/~47°), better for full-tabletop visibility.
    # The previous "Tensor sizes: [0]" errors during testing were SIGKILL
    # shutdown noise, not a config issue with this clone.
    head_camera: CameraCfg | None = None

    chest_left_camera: CameraCfg | None = None
    chest_right_camera: CameraCfg | None = None

    # --- Wrist cameras (per arm, exposed off the wrist body, see full hand) ----
    left_wrist_camera: CameraCfg | None = None
    right_wrist_camera: CameraCfg | None = None


##
# Actions
##
@configclass
class ActionsCfg:
    """Action: Pink IK for both arms + direct hand joint positions."""

    upper_body_ik = PinkInverseKinematicsActionCfg(
        # Waist deliberately NOT in pink_controlled_joint_names — matches GR1T2:
        # Pink IK does not move waist directly. Instead the NullSpacePostureTask
        # below holds it at zero via the regularizer.
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
            num_hand_joints=12,  # 6 per hand × 2 hands (full s6_hand, not proxy)
            show_ik_warnings=False,
            fail_on_joint_limit_violation=False,
            variable_input_tasks=[
                # IK weights copied from GR1T2 (pickplace_gr1t2_env_cfg.py):
                #   FrameTask: position_cost=8.0, orientation_cost=1.0, lm_damping=12, gain=0.5
                #   DampingTask: cost=0.5
                #   NullSpacePostureTask: cost=0.5, lm_damping=1, regularizing
                #     shoulders + elbows + waist toward zero.
                FrameTask(
                    f"{PINK_IK_FRAME_PREFIX}{LEFT_EEF_LINK}",
                    position_cost=8.0,
                    orientation_cost=1.0,
                    lm_damping=12,
                    gain=1.0,
                ),
                FrameTask(
                    f"{PINK_IK_FRAME_PREFIX}{RIGHT_EEF_LINK}",
                    position_cost=8.0,
                    orientation_cost=1.0,
                    lm_damping=12,
                    gain=1.0,
                ),
                DampingTask(cost=0.5),
                # Anchors the elbow-swivel null-space DOF so the shoulder
                # doesn't swing during reaches. Mask is shoulders (joints
                # 1-3) only — elbow and wrist stay free, waist is excluded
                # so Pink IK can use it for reach without fighting the
                # regularizer.
                NullSpacePostureTask(
                    cost=4.0,
                    lm_damping=1,
                    controlled_frames=[
                        f"{PINK_IK_FRAME_PREFIX}{LEFT_EEF_LINK}",
                        f"{PINK_IK_FRAME_PREFIX}{RIGHT_EEF_LINK}",
                    ],
                    # Match GR1T2's waist-handling pattern: waist_yaw is the
                    # only soft constraint pulling waist toward init (since
                    # waist was removed from pink_controlled_joint_names, Pink
                    # IK doesn't drive it directly). joint1 (shoulder swing)
                    # is also in the mask to damp the elbow-swivel null-space
                    # DOF that otherwise causes shoulder swing during reaches.
                    controlled_joints=(
                        [_LEFT_ARM_JOINTS[0], _LEFT_ARM_JOINTS[1],
                         _RIGHT_ARM_JOINTS[0], _RIGHT_ARM_JOINTS[1]]
                        + _WAIST_JOINTS
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

        # Cameras disabled — set to None so observation manager skips them (term_cfg is None → continue)
        head_camera_rgb: ObsTerm | None = None
        head_camera_depth: ObsTerm | None = None
        chest_left_camera_rgb: ObsTerm | None = None
        chest_right_camera_rgb: ObsTerm | None = None
        left_wrist_camera_rgb: ObsTerm | None = None
        right_wrist_camera_rgb: ObsTerm | None = None

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
        func=mdp.object_dropped_after_lift,
        params={
            "minimum_height": 0.5,
            "asset_cfg": SceneEntityCfg("object"),
            "robot_cfg": SceneEntityCfg("robot"),
            "table_top_z": 1.00,
            "lift_threshold_m": 0.05,
            "grasp_threshold": 0.20,
            "lift_hold_time_s": 0.5,
            "hand_body_names": ("left_arm_link07", "right_arm_link07"),
            "release_distance_m": 0.15,
        },
    )

    # Live-env success: position-on-tray + lift latch + hands-released.
    # The can must:
    #   - be inside the tray xy footprint (xy_threshold=0.10 m, strict),
    #   - be resting on the tray's actual top surface (z within
    #     [tray_top_z, tray_top_z + 0.10 m]),
    #   - have been lifted ≥5 cm above the table top (1.00 m) at some
    #     prior step in this episode (the "pick" moment),
    #   - and BOTH hand bodies (left_arm_link07, right_arm_link07) be at
    #     least 15 cm away from the can right now (the user has let go).
    success = DoneTerm(
        func=mdp.place_on_tray_with_lift,
        params={
            "object_a_cfg": SceneEntityCfg("object"),
            "robot_cfg": SceneEntityCfg("robot"),
            "xy_threshold": 0.10,
            "surface_band_low": 0.00,
            "surface_band_high": 0.10,
            "table_top_z": 1.00,
            "lift_threshold_m": 0.05,
            "hand_body_names": ("left_arm_link07", "right_arm_link07"),
            "release_distance_m": 0.15,
            "grasp_threshold": 0.20,
            "lift_hold_time_s": 0.5,
        },
    )


##
# Events
##
def _reset_object_off_tray(
    env,
    env_ids,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    side_offset_m: float = 0.30,
    jitter_m: float = 0.05,
):
    """Reset the can to a "safe zone" on the RIGHT side of the tray.

    Sets the can's xy to (env_origin_x + side_offset_m + U[-jitter,+jitter],
    env_origin_y + default_y + U[-jitter,+jitter]). z and orientation come
    from default_root_state.

    side_offset_m=0.30 puts the can ≥30cm right of env-origin (≈ tray center)
    with ±jitter wiggle. Tray half-width is ~13cm, so the can can never
    spawn within tray xy."""
    asset = env.scene[asset_cfg.name]
    root_states = asset.data.default_root_state[env_ids].clone()
    n = len(env_ids)
    jitter = (torch.rand((n, 2), device=asset.device, dtype=root_states.dtype) * 2.0 - 1.0) * jitter_m
    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids]
    positions[:, 0] = env.scene.env_origins[env_ids][:, 0] + side_offset_m + jitter[:, 0]
    positions[:, 1] = positions[:, 1] + jitter[:, 1]
    orientations = root_states[:, 3:7]
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.zeros((n, 6), device=asset.device), env_ids=env_ids)


@configclass
class EventCfg:
    # Runs once at env creation — hides the plastic crate / packaged props that
    # ship inside the PackingTable USD so the table surface is clean.
    hide_crate = EventTerm(func=_hide_packing_table_crate, mode="startup")

    # Startup-time query of the tray's world AABB max.z. Stashed on env so the
    # success function can use the actual tray surface height for the landing
    # check instead of guessing.
    compute_tray_top_z = EventTerm(func=_compute_tray_top_z, mode="startup")

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # Place can on a random side (L or R) of the fixed tray, with ±5cm jitter.
    # Side-offset 30cm guarantees the can never lands inside the tray xy
    # footprint (tray half-width ~13cm).
    reset_object = EventTerm(
        func=_reset_object_off_tray,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "side_offset_m": 0.30,
            "jitter_m": 0.05,
        },
    )

    # Per-reset color randomization for visual diversity.
    randomize_colors = EventTerm(
        func=_randomize_can_tray_colors,
        mode="reset",
    )

    # Tray reset event removed — tray init_state is the fixed pose.


##
# Env cfg
##
@configclass
class PickPlaceA2EnvCfg(ManagerBasedRLEnvCfg):
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

    # s6_hand grasp proxy: R_thumb_1_joint + R_index_1_joint as 2-joint stand-in for
    # the parallel-gripper check in place_after_grasp / object_a_is_into_b.
    gripper_joint_names = ["R_thumb_1_joint", "R_index_1_joint"]
    gripper_open_val = 0.0
    gripper_threshold = 0.5

    # Idle action: left wrist pose (7) + right wrist pose (7) + len(_HAND_JOINTS) hand joints (0 = open).
    # Computed in WORLD frame via pinocchio FK on the init joint pose above, so
    # Pink IK isn't dragging the wrists away from where they spawn each tick.
    # Regenerate when the init joint pose changes (or when the robot's spawn
    # pose changes from pos=(0, 0.05, 0.93), rot=(0.7071, 0, 0, 0.7071)).
    idle_action = torch.tensor(
        [
            -0.309326, 0.012184, 0.776431, 0.787864, 0.000621, -0.615840, -0.000830,
             0.309324, 0.012172, 0.776431, 0.787945, 0.000633,  0.615736,  0.000816,
            *([0.0] * len(_HAND_JOINTS)),
        ]
    )

    def __post_init__(self):
        self.decimation = 6
        self.episode_length_s = 20.0
        self.sim.dt = 1 / 120
        self.sim.render_interval = 2  # match GR1T2 — halves render cost, freeing GPU for tracking

        # Pink IK URDF: 41-joint patched URDF where 12 mimic-children (*_2,
        # *_3 finger joints) are flipped to fixed. Paired with the 41-joint
        # A2_nomimic_physics.usd (the .bad_today/.bak version). Both sides
        # must agree on joint count, or pink_ik.py:159 raises
        # ValueError('L_index_2_joint is not in list').
        self.actions.upper_body_ik.controller.urdf_path = (
            "/home/wagner/code/IsaacLab/assets/A2/A2_nomimic_for_pinkik.urdf"
        )
        # Mesh paths inside the URDF are relative (meshes/...), so the mesh root
        # is the parent of the meshes/ dir — i.e. the assets/A2 dir itself.
        self.actions.upper_body_ik.controller.mesh_path = "/home/wagner/code/IsaacLab/assets/A2"

        # Pull idle wrist poses straight from idle_action so the keyboard device
        # starts at a pose Pink IK already considers feasible.
        idle = self.idle_action.tolist()
        left_idle_pose = tuple(idle[0:7])
        right_idle_pose = tuple(idle[7:14])

        self.teleop_devices = DevicesCfg(
            devices={
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        A2RetargeterCfg(
                            enable_visualization=True,
                            num_open_xr_hand_joints=2 * self.NUM_OPENXR_HAND_JOINTS,
                            sim_device=self.sim.device,
                            hand_joint_names=_HAND_JOINTS,
                            # Persisted manual calibration — auto-cal disabled
                            # (auto_calibrate_seconds defaults to 0.0). The
                            # values below override anything the live retargeter
                            # would compute and stay fixed across sessions.
                            # Left side mirrors only the X-roll component of
                            # the right's compensation (right joint7 axis is
                            # 0 0 -1 in the URDF while left is 0 0 1, so the
                            # wrist-link-to-visible-hand offset has the
                            # opposite roll sign per side; yaw stays the same).
                            # Verified visually with Gemma: prior values
                            # (-1.54, 0, 1.54) over-mirrored and left the hand
                            # still rolled ~90-180° off.
                            left_fixed_rpy=(-1.54, 0, -1.54),
                            right_fixed_rpy=(1.54, 0, -1.54),
                            left_shoulder_pos=(-0.2, 0.05, 1.1),
                            right_shoulder_pos=(0.2, 0.05, 1.1),
                            # Stretch reach so the user's hand-forward motion
                            # maps to a deeper robot reach. pos_scale 1.0→1.3
                            # multiplies the user→robot offset from the shoulder
                            # anchor; max_reach 0.7→0.9 widens the clip sphere
                            # so the scaled offsets aren't truncated.
                            pos_scale=1.3,
                            max_reach=0.9,
                            debug_every=30,
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
                "keyboard": A2KeyboardCfg(sim_device=self.sim.device, hand_joint_names=_HAND_JOINTS),
            }
        )
