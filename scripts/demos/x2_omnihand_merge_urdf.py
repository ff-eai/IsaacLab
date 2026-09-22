"""Merge X2 ultra body URDF with left/right OmniHand URDFs into a single URDF.

Output: assets/X2_omnihand/X2_omnihand.urdf

Attachment points: left_wrist_roll_link / right_wrist_roll_link.
Mesh paths are rewritten to local layout:
    x2 body STLs    -> meshes/x2/<file>.STL
    omnihand STLs   -> meshes/omnihand/<file>.STL
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path("/home/wagner/code/IsaacLab")
SRC_X2 = Path("/home/wagner/Downloads/X2_URDF-v1.3.0/x2_ultra.urdf")
SRC_L  = Path("/home/wagner/Downloads/omnihand_description-omnihandT2_1/assets/urdf/omnihand_left.urdf")
SRC_R  = Path("/home/wagner/Downloads/omnihand_description-omnihandT2_1/assets/urdf/omnihand_right.urdf")
OUT    = REPO / "assets/X2_omnihand/X2_omnihand.urdf"


def _rewrite_mesh_paths(text: str, prefix: str) -> str:
    text = re.sub(r'filename="package://[^/]+/meshes/', f'filename="{prefix}', text)
    text = re.sub(r'filename="package://[^/]+/assets/meshes/', f'filename="{prefix}', text)
    text = re.sub(r'filename="\./meshes/', f'filename="{prefix}', text)
    return text


def _load_hand(path: Path, parent_link: str, palm_joint: str) -> list[ET.Element]:
    """Return omnihand links/joints to splice in, with palm parent rewired and meshes remapped."""
    text = _rewrite_mesh_paths(path.read_text(), prefix="meshes/omnihand/")
    root = ET.fromstring(text)
    out: list[ET.Element] = []
    for child in list(root):
        tag = child.tag
        if tag == "material":
            continue  # avoid duplicate material defs (x2 already declares them)
        if tag == "link" and child.get("name") == "base_link":
            continue  # drop omnihand's own root; we attach to x2 wrist
        if tag == "joint" and child.get("name") == palm_joint:
            # rewire palm's parent from omnihand base_link to x2 wrist link
            for p in child.findall("parent"):
                p.set("link", parent_link)
        out.append(child)
    return out


def _strip_visual_and_collision(robot_root: ET.Element, link_names: set[str]) -> None:
    """Remove all <visual> and <collision> children from the given links so the
    original X2 hand geometry baked into the wrist mesh doesn't render alongside
    the attached OmniHand."""
    for link in robot_root.findall("link"):
        if link.get("name") in link_names:
            for tag in ("visual", "collision"):
                for child in list(link.findall(tag)):
                    link.remove(child)


def _widen_joint_limits(robot_root: ET.Element, joint_names: set[str], lo: float, hi: float) -> None:
    """Override <limit lower=... upper=...> on the named joints — used to widen
    X2's narrow wrist limits (pitch ±0.558, roll asymmetric) so we can set any
    joint angle pre-teleop without hitting the IsaacLab limit-violation guard."""
    for joint in robot_root.findall("joint"):
        if joint.get("name") in joint_names:
            limit = joint.find("limit")
            if limit is not None:
                limit.set("lower", str(lo))
                limit.set("upper", str(hi))


def _flip_joint_axis(robot_root: ET.Element, joint_names: set[str]) -> None:
    """Negate <axis xyz=...> on the named joints. Used to reverse the rotation
    sense — same commanded value rotates the opposite direction."""
    for joint in robot_root.findall("joint"):
        if joint.get("name") in joint_names:
            axis = joint.find("axis")
            if axis is not None:
                xyz = [float(v) for v in axis.get("xyz", "0 0 0").split()]
                axis.set("xyz", " ".join(f"{-v}" for v in xyz))


def main() -> None:
    x2_text = _rewrite_mesh_paths(SRC_X2.read_text(), prefix="meshes/x2/")
    x2_root = ET.fromstring(x2_text)
    assert x2_root.tag == "robot"
    x2_root.set("name", "x2_omnihand")

    # Strip the original X2 hand geometry from the wrist links — the wrist_roll
    # mesh ships with a built-in hand, which renders alongside the attached
    # OmniHand and looks like two hands per arm.
    _strip_visual_and_collision(x2_root, {"left_wrist_roll_link", "right_wrist_roll_link"})

    # Per-side palm-joint rpy. Final tuned values for X2 + OmniHand mounting:
    # roll=0, pitch=π (180° pitch flip aligns OmniHand fingers with X2 forearm
    # axis), yaw=1.5 (~86° around Z, rotates palm-out to palm-down). Same
    # values both sides — preserves L/R mirror symmetry of the OmniHand URDFs.
    # OmniHand left and right URDFs are already mirror images of each other
    # (anatomically L vs R hand). Same palm-joint rpy on both produces correct
    # mirror behavior — different rpy breaks the mirror.
    for link_name, palm_joint, src, palm_rpy in [
        ("left_wrist_roll_link",  "L_palm_joint", SRC_L, (0.0, 3.1415927,  1.5707963)),
        ("right_wrist_roll_link", "R_palm_joint", SRC_R, (0.0, 3.1415927, -1.5707963)),
    ]:
        for el in _load_hand(src, link_name, palm_joint):
            if el.tag == "joint" and el.get("name") == palm_joint:
                origin = el.find("origin")
                if origin is not None:
                    origin.set("rpy", " ".join(f"{v}" for v in palm_rpy))
            x2_root.append(el)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(x2_root, space="  ")
    OUT.write_text('<?xml version="1.0"?>\n' + ET.tostring(x2_root, encoding="unicode") + "\n")
    n_joints = len(x2_root.findall("joint"))
    n_links = len(x2_root.findall("link"))
    print(f"wrote {OUT}  joints={n_joints}  links={n_links}")


if __name__ == "__main__":
    main()
