"""Build A2 + OmniHand T2 combined URDF.

Reads the existing expanded A2 URDF and replaces the s6_hand subtree on each
side with the OmniHand T2 (10-DOF-per-hand) URDF from the user's download.
Also copies OmniHand meshes into assets/A2_omnihand/meshes/omnihand/ and
rewrites `package://` references to relative paths rooted at A2_omnihand/.

Outputs:
  assets/A2_omnihand/A2_omnihand.urdf           (keeps <mimic> directives)
  assets/A2_omnihand/A2_omnihand_nomimic.urdf   (mimic stripped, for Pink IK/dex)
  assets/A2_omnihand/meshes/public/...          (symlink to A2 body meshes)
  assets/A2_omnihand/meshes/omnihand/...        (copy of OmniHand meshes)
"""

from __future__ import annotations

import copy
import os
import shutil
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path("/home/wagner/code/IsaacLab/assets")
SRC_A2 = ROOT / "A2" / "A2_expanded.urdf"
SRC_OMNIHAND = Path(
    "/home/wagner/Downloads/omni/1769938867609039/omnihand_description-omnihandT2_1/assets"
)
OUT_DIR = ROOT / "A2_omnihand"
OUT_URDF = OUT_DIR / "A2_omnihand.urdf"
OUT_NOMIMIC = OUT_DIR / "A2_omnihand_nomimic.urdf"


def _is_hand_element(elem: ET.Element) -> bool:
    """True if this <link>/<joint> belongs to the old s6_hand subtree."""
    name = elem.attrib.get("name", "")
    if name in {"left_hand", "right_hand"}:
        return True
    if name in {"left_arm_joint07_fixed", "right_arm_joint07_fixed"}:
        return True
    if name.startswith(("L_thumb", "R_thumb", "L_index", "R_index", "L_middle", "R_middle",
                        "L_ring", "R_ring", "L_pinky", "R_pinky")):
        return True
    return False


def _prepare_mesh_root() -> None:
    """Mirror A2 body meshes and copy OmniHand meshes into OUT_DIR/meshes/."""
    (OUT_DIR / "meshes").mkdir(parents=True, exist_ok=True)
    # Symlink A2 body meshes so they resolve via relative paths.
    src_public = ROOT / "A2" / "meshes" / "public"
    dst_public = OUT_DIR / "meshes" / "public"
    if dst_public.exists() or dst_public.is_symlink():
        dst_public.unlink() if dst_public.is_symlink() else shutil.rmtree(dst_public)
    dst_public.symlink_to(src_public, target_is_directory=True)
    # Copy OmniHand meshes (STLs only — skip collision subdir for now).
    dst_omni = OUT_DIR / "meshes" / "omnihand"
    if dst_omni.exists():
        shutil.rmtree(dst_omni)
    dst_omni.mkdir()
    src_omni = SRC_OMNIHAND / "meshes"
    for stl in src_omni.iterdir():
        if stl.is_file() and stl.suffix.upper() == ".STL":
            shutil.copy2(stl, dst_omni / stl.name)
    # Also copy collision meshes if present.
    col_src = src_omni / "collision"
    if col_src.exists():
        (dst_omni / "collision").mkdir(exist_ok=True)
        for stl in col_src.iterdir():
            if stl.is_file() and stl.suffix.upper() == ".STL":
                shutil.copy2(stl, dst_omni / "collision" / stl.name)


def _fixup_mesh_refs(root: ET.Element, *, prefix: str = "meshes/omnihand/") -> None:
    """Rewrite package://omnihand_description/meshes/... → meshes/omnihand/..."""
    for mesh in root.iter("mesh"):
        fn = mesh.attrib.get("filename", "")
        if fn.startswith("package://omnihand_description/meshes/"):
            mesh.set("filename", fn.replace(
                "package://omnihand_description/meshes/", prefix))


def _rename_base_link(root: ET.Element, side: str) -> str:
    """Rename OmniHand's `base_link` to `{side}_hand_base` so L and R don't collide.
    Returns the new name.
    """
    new_name = f"{side}_hand_base"
    for elem in list(root):
        if elem.tag == "link" and elem.attrib.get("name") == "base_link":
            elem.set("name", new_name)
        if elem.tag == "joint":
            parent = elem.find("parent")
            if parent is not None and parent.attrib.get("link") == "base_link":
                parent.set("link", new_name)
            child = elem.find("child")
            if child is not None and child.attrib.get("link") == "base_link":
                child.set("link", new_name)
    return new_name


def _make_attach_joint(side: str, hand_root_link: str, rpy: str) -> ET.Element:
    """A2 arm_link07 → OmniHand root link fixed joint."""
    j = ET.Element("joint", attrib={
        "name": f"{'left' if side == 'L' else 'right'}_arm_joint07_fixed",
        "type": "fixed",
    })
    ET.SubElement(j, "origin", attrib={"rpy": rpy, "xyz": "0 0 0"})
    ET.SubElement(j, "parent", attrib={"link": f"{'left' if side == 'L' else 'right'}_arm_link07"})
    ET.SubElement(j, "child", attrib={"link": hand_root_link})
    return j


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _prepare_mesh_root()

    a2_tree = ET.parse(str(SRC_A2))
    a2_root = a2_tree.getroot()

    # Drop s6_hand links + joints + attach-joint.
    for child in list(a2_root):
        if child.tag in ("link", "joint") and _is_hand_element(child):
            a2_root.remove(child)

    # Load and merge each OmniHand side.
    for side, urdf_name in (("L", "omnihand_left.urdf"), ("R", "omnihand_right.urdf")):
        omni_tree = ET.parse(str(SRC_OMNIHAND / "urdf" / urdf_name))
        omni_root = omni_tree.getroot()
        _fixup_mesh_refs(omni_root)
        hand_root = _rename_base_link(omni_root, side)

        # Skip top-level <material> tags — A2 URDF already has its own set,
        # and duplicates will fight. OmniHand materials are minimal anyway.
        for elem in list(omni_root):
            if elem.tag == "material":
                continue
            a2_root.append(copy.deepcopy(elem))

        # Match X2's OmniHand mounting rpy (per scripts/demos/x2_omnihand_merge_urdf.py):
        #   L: (roll=0, pitch=π, yaw=+π/2)   R: (roll=0, pitch=π, yaw=−π/2)
        # 180° pitch flip aligns OmniHand fingers with the forearm axis; ±π/2
        # yaw rotates palm-out to palm-down. Same values both sides preserves
        # the L/R mirror symmetry of the OmniHand URDFs.
        rpy = "0 3.1415927 1.5707963" if side == "L" else "0 3.1415927 -1.5707963"
        a2_root.append(_make_attach_joint(side, hand_root, rpy))

    # Write mimic-preserving output.
    ET.indent(a2_tree, space="  ")
    a2_tree.write(str(OUT_URDF), encoding="utf-8", xml_declaration=True)

    # Write nomimic variant.
    nomimic_tree = ET.parse(str(OUT_URDF))
    nomimic_root = nomimic_tree.getroot()
    for joint in nomimic_root.iter("joint"):
        m = joint.find("mimic")
        if m is not None:
            joint.remove(m)
    nomimic_tree.write(str(OUT_NOMIMIC), encoding="utf-8", xml_declaration=True)

    # Summary stats.
    n_links = sum(1 for _ in a2_root.iter("link"))
    n_joints = sum(1 for _ in a2_root.iter("joint"))
    n_rev = sum(1 for j in a2_root.iter("joint") if j.attrib.get("type") == "revolute")
    n_mimic = sum(1 for j in a2_root.iter("joint") if j.find("mimic") is not None)
    print(f"[build_a2_omnihand] wrote {OUT_URDF}")
    print(f"    links={n_links}  joints={n_joints}  revolute={n_rev}  mimic={n_mimic}")
    print(f"[build_a2_omnihand] wrote {OUT_NOMIMIC} (mimic stripped)")


if __name__ == "__main__":
    main()
