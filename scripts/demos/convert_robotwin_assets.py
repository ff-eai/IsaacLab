"""Convert RoboTwin GLB meshes → USD for Isaac Lab. Uses trimesh to load the GLB
(handles textures, multiple meshes) and writes the geometry directly into a USD
stage as UsdGeomMesh prims. Also uses the matching collision mesh so physics is
accurate.

Outputs under /home/wagner/code/IsaacLab/assets/robotwin/:
    071_can_base{0..6}.usd    (visual + collision + material)
    008_tray_base{0..3}.usd

Run once:
    ./isaaclab.sh -p scripts/demos/convert_robotwin_assets.py --headless
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app = AppLauncher(args_cli).app

import numpy as np
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

SRC_ROOT = "/home/wagner/code/RoboTwin/assets/objects"
OUT_ROOT = "/home/wagner/code/IsaacLab/assets/robotwin"


def load_glb_as_tri(glb_path: str) -> trimesh.Trimesh:
    """Load a GLB and return a concatenated Trimesh (merges multi-mesh scenes)."""
    obj = trimesh.load(glb_path, force="mesh", process=False)
    if isinstance(obj, trimesh.Scene):
        obj = obj.dump(concatenate=True)
    return obj


def write_mesh_to_usd_prim(stage: Usd.Stage, prim_path: str, mesh: trimesh.Trimesh, scale: float = 1.0) -> Usd.Prim:
    """Write a trimesh as a UsdGeomMesh at the given prim path. Applies a uniform scale."""
    usd_mesh = UsdGeom.Mesh.Define(stage, prim_path)
    verts = np.asarray(mesh.vertices, dtype=np.float32) * scale
    faces = np.asarray(mesh.faces, dtype=np.int32)
    usd_mesh.CreatePointsAttr(verts.tolist())
    usd_mesh.CreateFaceVertexIndicesAttr(faces.flatten().tolist())
    usd_mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    # Compute extent (bounding box) required for USD meshes.
    mn = verts.min(axis=0).tolist() if len(verts) else [0, 0, 0]
    mx = verts.max(axis=0).tolist() if len(verts) else [0, 0, 0]
    usd_mesh.CreateExtentAttr([Gf.Vec3f(*mn), Gf.Vec3f(*mx)])
    return usd_mesh.GetPrim()


def convert(glb_visual: str, glb_collision: str | None, model_json: str | None, dst_usd: str) -> bool:
    if not os.path.isfile(glb_visual):
        print(f"[conv] missing {glb_visual}")
        return False

    os.makedirs(os.path.dirname(dst_usd), exist_ok=True)

    # Per-model scale from RoboTwin metadata (if present).
    scale = 1.0
    if model_json and os.path.isfile(model_json):
        try:
            meta = json.load(open(model_json))
            s = meta.get("scale", [1, 1, 1])
            scale = float(s[0]) if isinstance(s, (list, tuple)) else float(s)
        except Exception:
            pass

    try:
        visual_tri = load_glb_as_tri(glb_visual)
    except Exception as e:
        print(f"[conv] FAIL load visual: {e}")
        return False

    collision_tri = None
    if glb_collision and os.path.isfile(glb_collision):
        try:
            collision_tri = load_glb_as_tri(glb_collision)
        except Exception:
            collision_tri = None

    # Build USD stage.
    stage = Usd.Stage.CreateNew(dst_usd)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root_xform = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root_xform.GetPrim())

    # Apply RigidBodyAPI + MassAPI on the root so Isaac Lab's RigidObjectCfg
    # can discover this asset as a rigid body.
    UsdPhysics.RigidBodyAPI.Apply(root_xform.GetPrim())
    UsdPhysics.MassAPI.Apply(root_xform.GetPrim())

    # Visual mesh at /World/visuals/mesh, collision at /World/collisions/mesh
    UsdGeom.Xform.Define(stage, "/World/visuals")
    visual_prim = write_mesh_to_usd_prim(stage, "/World/visuals/mesh", visual_tri, scale=scale)

    UsdGeom.Xform.Define(stage, "/World/collisions")
    collision_source = collision_tri if collision_tri is not None else visual_tri
    collision_prim = write_mesh_to_usd_prim(stage, "/World/collisions/mesh", collision_source, scale=scale)

    # Tag collision mesh with physics collision API + convex decomposition.
    UsdPhysics.CollisionAPI.Apply(collision_prim)
    mesh_coll = UsdPhysics.MeshCollisionAPI.Apply(collision_prim)
    mesh_coll.CreateApproximationAttr().Set("convexDecomposition")
    # Hide the collision mesh so only visuals render.
    UsdGeom.Imageable(collision_prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)

    stage.GetRootLayer().Save()
    return True


def main() -> None:
    os.makedirs(OUT_ROOT, exist_ok=True)
    total_ok = 0
    total_fail = 0
    for modelname in ("071_can", "008_tray", "001_bottle"):
        src_root = os.path.join(SRC_ROOT, modelname)
        visual_dir = os.path.join(src_root, "visual")
        collision_dir = os.path.join(src_root, "collision")
        if not os.path.isdir(visual_dir):
            print(f"[skip] {visual_dir}")
            continue
        for fname in sorted(os.listdir(visual_dir)):
            if not fname.endswith(".glb"):
                continue
            base = fname.replace(".glb", "")
            src_vis = os.path.join(visual_dir, fname)
            src_col = os.path.join(collision_dir, fname) if os.path.isdir(collision_dir) else None
            meta = os.path.join(src_root, f"model_data{base.replace('base', '')}.json")
            dst = os.path.join(OUT_ROOT, f"{modelname}_{base}.usd")
            print(f"[conv] {src_vis}  →  {dst}")
            if convert(src_vis, src_col, meta, dst):
                total_ok += 1
            else:
                total_fail += 1

    print(f"[conv] done — {total_ok} succeeded, {total_fail} failed")


main()
app.close()
