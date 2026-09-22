"""Configuration for X2 humanoid with simple grasper hands.

This module builds a merged URDF on the fly by:
- Taking the existing X2+OmniHand URDF (for the X2 body links/joints)
- Removing the OmniHand sections
- Attaching a small grasper URDF (from ~/Downloads/robot_description) to each wrist

The resulting URDF is spawned directly (URDF->USD conversion happens at runtime via UrdfFileCfg).
"""

from __future__ import annotations

import os
import re
import tempfile

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
)

_X2_OMNIHAND_URDF = os.path.join(_REPO_ROOT, "assets", "X2_omnihand", "X2_omnihand.urdf")
_GRASPER_XACRO = os.path.expanduser("~/Downloads/robot_description/urdf/robot_urdf.xacro")
_GRASPER_MESH_DIR = os.path.expanduser("~/Downloads/robot_description/meshes")

# Mount rotation applied at the wrist->grasper fixed joint.
# If the grasper appears to point "into" the forearm, adjust these.
_L_GRASPER_MOUNT_RPY = (0.0, 3.1415927, 0)  # (roll, pitch, yaw)
_R_GRASPER_MOUNT_RPY = (0.0, 3.1415927, 0)


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _extract_grasper_urdf_body(xacro_text: str) -> str:
    """Extract the URDF snippet inside the xacro macro, excluding xacro headers."""
    # Keep everything from the first <link ...> to the end of the macro.
    start = xacro_text.find("<link")
    end = xacro_text.rfind("</xacro:macro>")
    if start < 0 or end < 0 or end <= start:
        raise ValueError("Could not locate URDF body inside grasper xacro.")
    body = xacro_text[start:end]
    # Drop the xacro property line if it remained (defensive).
    body = re.sub(r"^\s*<xacro:property[^>]+/>\s*$", "", body, flags=re.MULTILINE)
    return body.strip() + "\n"


def _prefix_grasper(urdf_snippet: str, prefix: str) -> str:
    """Prefix all link/joint names in the grasper snippet and rewrite mesh URLs."""
    # Replace mesh references with absolute file:// URIs.
    mesh_dir_uri = f"file://{_GRASPER_MESH_DIR}"
    out = urdf_snippet.replace("package://robot_description/meshes", mesh_dir_uri)
    out = out.replace("${mesh_path}", mesh_dir_uri)

    # Known link/joint names in the provided xacro.
    links = [
        "base_link",
        "hand_narrow1_Link",
        "hand_narrow2_Link",
        "hand_narrow3_Link",
        "hand_narrow4_Link",
        "hand_narrow_loop_Link",
        "hand_wide1_Link",
        "hand_wide2_Link",
        "hand_wide3_Link",
        "hand_wide4_Link",
        "hand_wide_loop_Link",
    ]
    joints = [
        "hand_narrow1_joint",
        "hand_narrow2_joint",
        "hand_narrow3_joint",
        "hand_narrow4_joint",
        "hand_narrow_loop_joint",
        "hand_wide1_joint",
        "hand_wide2_joint",
        "hand_wide3_joint",
        "hand_wide4_joint",
        "hand_wide_loop_joint",
    ]

    # Base link gets a clearer name so downstream code can refer to it.
    base_repl = f"{prefix}grasper_base_link"
    out = out.replace('name="base_link"', f'name="{base_repl}"')
    out = out.replace('link="base_link"', f'link="{base_repl}"')

    # Prefix the remaining names.
    for name in links:
        if name == "base_link":
            continue
        out = out.replace(f'name="{name}"', f'name="{prefix}{name}"')
        out = out.replace(f'link="{name}"', f'link="{prefix}{name}"')
    for name in joints:
        out = out.replace(f'name="{name}"', f'name="{prefix}{name}"')
        # No 'link=' occurrences for joints.

    return out.strip() + "\n"


def _build_x2_grasper_urdf(output_path: str) -> str:
    """Write merged X2+grasper URDF and return output_path."""
    if not os.path.isfile(_X2_OMNIHAND_URDF):
        raise FileNotFoundError(f"Missing base X2 URDF: {_X2_OMNIHAND_URDF}")
    if not os.path.isfile(_GRASPER_XACRO):
        raise FileNotFoundError(f"Missing grasper xacro: {_GRASPER_XACRO}")
    if not os.path.isdir(_GRASPER_MESH_DIR):
        raise FileNotFoundError(f"Missing grasper mesh dir: {_GRASPER_MESH_DIR}")

    x2_text = _read_text(_X2_OMNIHAND_URDF)
    xacro_text = _read_text(_GRASPER_XACRO)

    # Strip the OmniHand part: it starts at L_palm_joint in this repo's merged URDF.
    cut = x2_text.find('<joint name="L_palm_joint"')
    if cut < 0:
        raise ValueError("Could not find L_palm_joint in X2_omnihand URDF; cannot strip hands.")
    x2_prefix = x2_text[:cut].rstrip() + "\n"
    # Rewrite X2 body meshes to absolute file:// URIs so URDF->USD conversion can
    # find them even though we write the merged URDF under /tmp.
    x2_mesh_root_uri = f"file://{os.path.join(_REPO_ROOT, 'assets', 'X2_omnihand')}"
    x2_prefix = x2_prefix.replace('filename="meshes/', f'filename="{x2_mesh_root_uri}/meshes/')

    grasper_body = _extract_grasper_urdf_body(xacro_text)
    left = _prefix_grasper(grasper_body, "L_")
    right = _prefix_grasper(grasper_body, "R_")

    merged = []
    merged.append(x2_prefix)
    merged.append('  <!-- Attached graspers (generated from ~/Downloads/robot_description) -->\n')
    merged.append('  <joint name="L_grasper_mount_joint" type="fixed">\n')
    merged.append(
        f'    <origin xyz="0 0 0" rpy="{_L_GRASPER_MOUNT_RPY[0]} {_L_GRASPER_MOUNT_RPY[1]} {_L_GRASPER_MOUNT_RPY[2]}" />\n'
    )
    merged.append('    <parent link="left_wrist_roll_link" />\n')
    merged.append('    <child link="L_grasper_base_link" />\n')
    merged.append('  </joint>\n')
    merged.append('  <joint name="R_grasper_mount_joint" type="fixed">\n')
    merged.append(
        f'    <origin xyz="0 0 0" rpy="{_R_GRASPER_MOUNT_RPY[0]} {_R_GRASPER_MOUNT_RPY[1]} {_R_GRASPER_MOUNT_RPY[2]}" />\n'
    )
    merged.append('    <parent link="right_wrist_roll_link" />\n')
    merged.append('    <child link="R_grasper_base_link" />\n')
    merged.append('  </joint>\n\n')
    merged.append(left)
    merged.append("\n")
    merged.append(right)
    merged.append("\n</robot>\n")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("".join(merged))

    return output_path


def _x2_grasper_actuators() -> dict:
    """Reuse X2 arm/waist tuning; add PD for the grasper joints."""
    return {
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"],
            velocity_limit_sim=None,
            effort_limit_sim=None,
            stiffness=None,
            damping=None,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_.*_joint"],
            velocity_limit_sim=None,
            effort_limit_sim=None,
            stiffness=4400.0,
            damping=40.0,
            armature=0.01,
        ),
        "left_arm": ImplicitActuatorCfg(
            joint_names_expr=["left_shoulder_.*", "left_elbow_joint", "left_wrist_.*"],
            velocity_limit_sim=None,
            effort_limit_sim=None,
            stiffness=4400.0,
            damping=40.0,
            armature=0.01,
        ),
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=["right_shoulder_.*", "right_elbow_joint", "right_wrist_.*"],
            velocity_limit_sim=None,
            effort_limit_sim=None,
            stiffness=4400.0,
            damping=40.0,
            armature=0.01,
        ),
        # Grasper joints come from robot_description xacro.
        "left_grasper": ImplicitActuatorCfg(
            joint_names_expr=["L_hand_.*_joint"],
            velocity_limit_sim=None,
            effort_limit_sim=None,
            stiffness=200.0,
            damping=5.0,
        ),
        "right_grasper": ImplicitActuatorCfg(
            joint_names_expr=["R_hand_.*_joint"],
            velocity_limit_sim=None,
            effort_limit_sim=None,
            stiffness=200.0,
            damping=5.0,
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_.*_joint"],
            velocity_limit_sim=None,
            effort_limit_sim=None,
            stiffness=None,
            damping=None,
        ),
    }


def _get_generated_urdf_path() -> str:
    # Stable path so the URDF->USD converter can cache results.
    out_dir = os.path.join(tempfile.gettempdir(), "isaaclab_x2_grasper")
    out_path = os.path.join(out_dir, "X2_grasper.generated.urdf")
    if not os.path.isfile(out_path):
        _build_x2_grasper_urdf(out_path)
    return out_path


X2_GRASPER_URDF_PATH = _get_generated_urdf_path()
X2_GRASPER_MESH_PATH = os.path.join(_REPO_ROOT, "assets", "X2_omnihand")


X2_GRASPER_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=X2_GRASPER_URDF_PATH,
        fix_base=False,
        force_usd_conversion=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            # If the imported collision geometry starts in slight penetration,
            # PhysX can "pop" the robot upward. Keep depenetration conservative.
            max_depenetration_velocity=0.3,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    # Default spawn height; pickplace_x2_grasper_env_cfg overrides with scene-tuned pose.
    init_state=ArticulationCfg.InitialStateCfg(joint_pos={".*": 0.0}, pos=(0.0, 0.0, 0.607)),
    actuators=_x2_grasper_actuators(),
)

