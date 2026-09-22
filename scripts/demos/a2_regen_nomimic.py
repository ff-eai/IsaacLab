"""Regenerate A2 USD with mimic joints converted to normal joints, and extract
per-hand URDFs (L_hand.urdf, R_hand.urdf) needed by the dex_retargeting library.

Outputs (all under assets/A2/):
  - A2_nomimic.usd               — fixes the PhysX hang by expanding mimic joints
  - A2_nomimic_converter_cfg/    — URDF converter cache dir
  - L_hand.urdf, R_hand.urdf     — per-hand URDFs for dex_retargeting

Run once:
    ./isaaclab.sh -p scripts/demos/a2_regen_nomimic.py --headless
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import xml.etree.ElementTree as ET

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app = AppLauncher(args_cli).app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

ASSETS_DIR = "/home/wagner/code/IsaacLab/assets/A2"
SRC_URDF = os.path.join(ASSETS_DIR, "A2.urdf")
STRIPPED_URDF = os.path.join(ASSETS_DIR, "A2_expanded.urdf")
OUT_USD_NAME = "A2_nomimic.usd"
OUT_USD_PATH = os.path.join(ASSETS_DIR, OUT_USD_NAME)


import re

# Links that form 4-bar closed-chain linkages on the real A2 (parallel foot rods
# and parallel wrist rods). PhysX articulation solver loops indefinitely on these,
# even when the joints are type="fixed". They aren't needed for sim-side
# manipulation, so strip them entirely. The remaining tree is a pure open chain.
_CLOSED_CHAIN_LINK_RE = re.compile(
    r"^(left|right)_(toe_[AB](_rod)?|wrist_(motor|rod)_[AB])$"
)


def _is_closed_chain_link(name: str) -> bool:
    return bool(_CLOSED_CHAIN_LINK_RE.match(name or ""))


def strip_mimic_and_closed_chains() -> None:
    """Produce a PhysX-friendly URDF:
      1. Strip <mimic> tags (controller imposes coupling downstream).
      2. Remove 4-bar closed-chain links + any joint touching them.
    """
    tree = ET.parse(SRC_URDF)
    robot = tree.getroot()

    # Pass 1: strip <mimic> tags.
    removed_mimic = 0
    for joint in robot.findall("joint"):
        mimic = joint.find("mimic")
        if mimic is not None:
            joint.remove(mimic)
            removed_mimic += 1

    # Pass 2: remove closed-chain links + joints that touch them (as parent or child).
    removed_links = []
    for link in list(robot.findall("link")):
        if _is_closed_chain_link(link.get("name", "")):
            robot.remove(link)
            removed_links.append(link.get("name"))

    removed_joints = []
    for joint in list(robot.findall("joint")):
        parent_el = joint.find("parent")
        child_el = joint.find("child")
        parent = parent_el.get("link") if parent_el is not None else ""
        child = child_el.get("link") if child_el is not None else ""
        if _is_closed_chain_link(parent) or _is_closed_chain_link(child):
            robot.remove(joint)
            removed_joints.append(joint.get("name"))

    tree.write(STRIPPED_URDF, encoding="utf-8", xml_declaration=True)
    print(f"[strip] mimic tags removed: {removed_mimic}")
    print(f"[strip] closed-chain links removed ({len(removed_links)}): {removed_links}")
    print(f"[strip] closed-chain joints removed ({len(removed_joints)}): {removed_joints}")
    print(f"[strip] output: {STRIPPED_URDF}")


def regen_usd() -> None:
    print(f"[regen] converting {STRIPPED_URDF} -> {OUT_USD_PATH} (mimic tags stripped)")
    cfg = UrdfConverterCfg(
        asset_path=STRIPPED_URDF,
        usd_dir=ASSETS_DIR,
        usd_file_name=OUT_USD_NAME,
        force_usd_conversion=True,
        make_instanceable=False,  # instanced prims block collision-property modification at runtime
        fix_base=True,
        root_link_name=None,
        link_density=0.0,
        merge_fixed_joints=True,
        convert_mimic_joints_to_normal_joints=True,  # key change — expand mimic to independent joints
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=100.0, damping=1.0),
        ),
        collider_type="convex_hull",
        self_collision=False,
        replace_cylinders_with_capsules=False,
        collision_from_visuals=False,
    )
    conv = UrdfConverter(cfg)
    print(f"[regen] output: {conv.usd_path}")
    # Verify destination.
    if not os.path.isfile(OUT_USD_PATH):
        alt = conv.usd_path
        if os.path.isfile(alt):
            print(f"[regen] copying {alt} -> {OUT_USD_PATH}")
            shutil.copy(alt, OUT_USD_PATH)
        else:
            sys.exit(f"[regen] FAIL: USD not produced at {OUT_USD_PATH} nor {alt}")


def extract_hand_urdf(side: str) -> None:
    """Write a URDF containing only the links/joints of one hand (s6_hand subtree).

    The dex_retargeting library wants a self-contained URDF whose root is the hand
    wrist. We stitch together the URDF header, the side-specific hand links/joints,
    and a synthetic root 'hand_base' that matches the 'wrist_link_name' in the YAML.
    """
    assert side in ("left", "right")
    prefix = "L" if side == "left" else "R"
    wrist_root = f"{side}_hand_base"  # synthetic root

    tree = ET.parse(SRC_URDF)
    robot = tree.getroot()

    # Find all hand-side links and joints (all start with L_ or R_).
    kept_links: dict[str, ET.Element] = {}
    kept_joints: list[ET.Element] = []
    for child in list(robot):
        name = child.get("name", "")
        if child.tag == "link":
            if name.startswith(f"{prefix}_"):
                kept_links[name] = child
        elif child.tag == "joint":
            if name.startswith(f"{prefix}_"):
                kept_joints.append(child)

    if not kept_joints:
        sys.exit(f"[extract] no {prefix}_ joints found in {SRC_URDF}")

    # Identify root link of this subtree — a link referenced as a child but whose
    # parent is not in kept_links. That parent is `left_hand` (removed here); we
    # replace it with a synthetic wrist_root.
    child_set = {j.find("child").get("link") for j in kept_joints if j.find("child") is not None}
    parent_set = {j.find("parent").get("link") for j in kept_joints if j.find("parent") is not None}
    external_parents = parent_set - set(kept_links.keys())
    # Typically just one external parent — the hand base.
    print(f"[extract] side={side}: {len(kept_links)} links, {len(kept_joints)} joints, "
          f"external parent(s): {external_parents}")

    # Build new URDF.
    new_robot = ET.Element("robot", attrib={"name": f"a2_{side}_hand"})

    # Synthetic wrist root link.
    wrist_link = ET.SubElement(new_robot, "link", attrib={"name": wrist_root})
    inertial = ET.SubElement(wrist_link, "inertial")
    ET.SubElement(inertial, "origin", attrib={"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(inertial, "mass", attrib={"value": "0.01"})
    ET.SubElement(
        inertial,
        "inertia",
        attrib={"ixx": "1e-6", "ixy": "0", "ixz": "0", "iyy": "1e-6", "iyz": "0", "izz": "1e-6"},
    )

    # Re-parent joints whose parent is in external_parents to the synthetic root.
    for j in kept_joints:
        parent_el = j.find("parent")
        if parent_el is not None and parent_el.get("link") in external_parents:
            parent_el.set("link", wrist_root)

    # Append all links and joints.
    for name, link in kept_links.items():
        new_robot.append(link)
    for j in kept_joints:
        new_robot.append(j)

    # Fix mesh paths to be absolute (dex_retargeting resolves relative to URDF dir
    # which is OK if we keep the URDF in assets/A2/).
    # We leave meshes untouched since the hand URDF will live in assets/A2/ alongside the meshes/ dir.

    out_path = os.path.join(ASSETS_DIR, f"{prefix}_hand.urdf")
    ET.ElementTree(new_robot).write(out_path, encoding="utf-8", xml_declaration=True)
    print(f"[extract] wrote {out_path}")


def main() -> None:
    if not os.path.isfile(SRC_URDF):
        sys.exit(f"source URDF not found: {SRC_URDF}")
    strip_mimic_and_closed_chains()
    regen_usd()
    extract_hand_urdf("left")
    extract_hand_urdf("right")
    print("[regen] done.")


if __name__ == "__main__":
    main()
    app.close()
