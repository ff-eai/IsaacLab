"""Produce hand-only OmniHand URDFs (with mimic) for dex-retargeting.

Dex-retargeting's URDF parser respects <mimic> joint directives and collapses
them into dependent variables — so a URDF with 10 driven + 6 mimic pip/dip
joints gives the DexPilot optimizer exactly 10 independent DOF, matching our
target_joint_names list.

Builds from the original OmniHand URDFs (which retain <mimic>), renames
`base_link` → `{L,R}_hand_base`, and rewrites `package://omnihand_description/
meshes/...` to the absolute path under assets/A2_omnihand/meshes/omnihand/.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from xml.etree import ElementTree as ET

SRC_OMNIHAND = Path(
    "/home/wagner/Downloads/omni/1769938867609039/omnihand_description-omnihandT2_1/assets"
)
ABS_MESHES = "/home/wagner/code/IsaacLab/assets/A2_omnihand/meshes/omnihand/"
OUT_DIR = Path("/home/wagner/code/IsaacLab/assets/A2_omnihand")


def _rename_base_link(root: ET.Element, side: str) -> None:
    new_name = f"{side}_hand_base"
    for elem in root:
        if elem.tag == "link" and elem.attrib.get("name") == "base_link":
            elem.set("name", new_name)
        if elem.tag == "joint":
            p = elem.find("parent")
            if p is not None and p.attrib.get("link") == "base_link":
                p.set("link", new_name)
            c = elem.find("child")
            if c is not None and c.attrib.get("link") == "base_link":
                c.set("link", new_name)


def _fix_mesh_paths(root: ET.Element) -> None:
    for mesh in root.iter("mesh"):
        fn = mesh.attrib.get("filename", "")
        if fn.startswith("package://omnihand_description/meshes/"):
            mesh.set("filename", fn.replace(
                "package://omnihand_description/meshes/", ABS_MESHES))


def main() -> None:
    for side, name in (("L", "omnihand_left.urdf"), ("R", "omnihand_right.urdf")):
        tree = ET.parse(str(SRC_OMNIHAND / "urdf" / name))
        root = tree.getroot()
        _rename_base_link(root, side)
        _fix_mesh_paths(root)
        # Keep the mimic tags — that's the whole point of this build.
        out = OUT_DIR / f"{side}_omnihand.urdf"
        ET.indent(tree, space="  ")
        tree.write(str(out), encoding="utf-8", xml_declaration=True)
        n_joints = sum(1 for _ in root.iter("joint"))
        n_mimic = sum(1 for j in root.iter("joint") if j.find("mimic") is not None)
        print(f"[build] {out}  joints={n_joints}  mimic={n_mimic}")


if __name__ == "__main__":
    main()
