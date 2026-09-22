"""Regenerate X2_omnihand.usd with merge_fixed_joints=True so the OmniHand
wrist-mount fixed-joint rotation is baked into link transforms.

Mirrors a2_omnihand_regen_nomimic.py — same UrdfConverterCfg, just pointed at
the X2 merged URDF (assets/X2_omnihand/X2_omnihand.urdf).

Run once:
    ./isaaclab.sh -p scripts/demos/x2_omnihand_regen_nomimic.py --headless
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app = AppLauncher(args_cli).app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

ASSETS_DIR = "/home/wagner/code/IsaacLab/assets/X2_omnihand"
SRC_URDF = os.path.join(ASSETS_DIR, "X2_omnihand.urdf")
OUT_USD_NAME = "X2_omnihand.usd"
OUT_USD_PATH = os.path.join(ASSETS_DIR, OUT_USD_NAME)


def regen_usd() -> None:
    print(f"[regen] converting {SRC_URDF} -> {OUT_USD_PATH}")
    cfg = UrdfConverterCfg(
        asset_path=SRC_URDF,
        usd_dir=ASSETS_DIR,
        usd_file_name=OUT_USD_NAME,
        force_usd_conversion=True,
        make_instanceable=False,
        fix_base=True,
        root_link_name=None,
        link_density=0.0,
        merge_fixed_joints=True,
        convert_mimic_joints_to_normal_joints=True,
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
    if not os.path.isfile(OUT_USD_PATH):
        alt = conv.usd_path
        if os.path.isfile(alt):
            print(f"[regen] copying {alt} -> {OUT_USD_PATH}")
            shutil.copy(alt, OUT_USD_PATH)
        else:
            sys.exit(f"[regen] FAIL: USD not produced at {OUT_USD_PATH} nor {alt}")


def main() -> None:
    if not os.path.isfile(SRC_URDF):
        sys.exit(f"source URDF not found: {SRC_URDF}")
    regen_usd()
    print("[regen] done.")


if __name__ == "__main__":
    main()
    app.close()
