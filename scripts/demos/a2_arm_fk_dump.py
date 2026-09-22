"""Pure-kinematic FK dump for A2 arm joints — no Isaac Sim, no policy server.

Loads the A2 URDF via pinocchio and computes the world-frame position of
``left_arm_link07`` and ``right_arm_link07`` (the EEF frames Pink IK targets)
for arbitrary 14-d arm joint settings. Use it to figure out which joint(s)
rotate the arm to a desired pose without running a full eval.

Usage:
    python scripts/demos/a2_arm_fk_dump.py            # T-pose baseline + sweep
    python scripts/demos/a2_arm_fk_dump.py --left j1=0,j2=1.57,j3=0,j4=0
    python scripts/demos/a2_arm_fk_dump.py --probe j2 --range -1.5,1.5,0.5
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

try:
    import pinocchio as pin
except ImportError:
    sys.exit("pinocchio required: use the isaaclab conda env")


URDF_PATH = "/home/wagner/code/IsaacLab/assets/A2/A2.urdf"
MESH_DIR = "/home/wagner/code/IsaacLab/assets/A2"

# Same joint name lists used by the eval script — order is L7 then R7.
LEFT_ARM_JOINT_NAMES = [f"idx{13 + i:02d}_left_arm_joint{i + 1}" for i in range(7)]
RIGHT_ARM_JOINT_NAMES = [f"idx{20 + i:02d}_right_arm_joint{i + 1}" for i in range(7)]
LEFT_EEF_LINK = "left_arm_link07"
RIGHT_EEF_LINK = "right_arm_link07"


def build_model() -> tuple[pin.Model, pin.Data]:
    model = pin.buildModelFromUrdf(URDF_PATH)
    data = model.createData()
    return model, data


def parse_joint_spec(spec: str) -> dict[int, float]:
    """Parse 'j2=1.57,j4=-0.10' into {1: 1.57, 3: -0.10} (0-indexed)."""
    out: dict[int, float] = {}
    if not spec:
        return out
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        name, value = tok.split("=")
        idx_1based = int(name.lstrip("j"))
        out[idx_1based - 1] = float(value)
    return out


def make_q(model: pin.Model, left_joints: dict[int, float], right_joints: dict[int, float]) -> np.ndarray:
    """Build the full configuration vector with all joints at 0 except specified arm joints."""
    q = pin.neutral(model)
    for joint_dict, names in (
        (left_joints, LEFT_ARM_JOINT_NAMES),
        (right_joints, RIGHT_ARM_JOINT_NAMES),
    ):
        for idx, val in joint_dict.items():
            jname = names[idx]
            jid = model.getJointId(jname)
            if jid >= len(model.idx_qs):
                continue
            q[model.idx_qs[jid]] = val
    return q


def fk_eef(model: pin.Model, data: pin.Data, q: np.ndarray, link: str) -> np.ndarray:
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    fid = model.getFrameId(link)
    return np.asarray(data.oMf[fid].translation, dtype=np.float64)


def report(model, data, label: str, left_joints=None, right_joints=None):
    q = make_q(model, left_joints or {}, right_joints or {})
    l_pos = fk_eef(model, data, q, LEFT_EEF_LINK)
    r_pos = fk_eef(model, data, q, RIGHT_EEF_LINK)
    print(f"  {label:50s}  L={l_pos.round(3).tolist()}  R={r_pos.round(3).tolist()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", type=str, default="", help="Left arm joints, e.g. 'j2=1.0,j4=-0.5'")
    ap.add_argument("--right", type=str, default="", help="Right arm joints (mirror with explicit signs).")
    ap.add_argument("--probe", type=str, default="", help="Single-joint probe, e.g. 'j2' (sweeps that joint)")
    ap.add_argument("--range", type=str, default="-1.5708,1.5708,0.5236",
                    help="lo,hi,step for --probe sweeps (default ±π/2 in 30° increments)")
    args = ap.parse_args()

    model, data = build_model()
    print(f"# A2 URDF arm FK — {model.nq}-dof free, {len(LEFT_ARM_JOINT_NAMES)} per arm")
    print(f"# All joints zeroed except those listed; positions in world frame (URDF root).")
    print()

    if args.probe:
        idx_1 = int(args.probe.lstrip("j"))
        lo, hi, step = (float(s) for s in args.range.split(","))
        print(f"# Probing left {LEFT_ARM_JOINT_NAMES[idx_1 - 1]} from {lo} to {hi} step {step}")
        v = lo
        report(model, data, "(reference, all 0)")
        while v <= hi + 1e-6:
            report(model, data, f"L j{idx_1}={v:+.3f} (only)", left_joints={idx_1 - 1: v})
            v += step
        print()
        print(f"# Probing right {RIGHT_ARM_JOINT_NAMES[idx_1 - 1]} from {lo} to {hi} step {step}")
        v = lo
        while v <= hi + 1e-6:
            report(model, data, f"R j{idx_1}={v:+.3f} (only)", right_joints={idx_1 - 1: v})
            v += step
        return

    if args.left or args.right:
        left = parse_joint_spec(args.left)
        right = parse_joint_spec(args.right)
        print(f"# User-specified pose:  left={left}  right={right}")
        report(model, data, "user pose", left_joints=left, right_joints=right)
        return

    # Default: dump common preset poses for visual reference.
    print("# Preset comparisons:")
    report(model, data, "T-pose (all 0)")
    report(model, data, "j2 = +π/2  (left only)", left_joints={1: 1.5708})
    report(model, data, "j2 = ±π/2  mirror",
           left_joints={1: 1.5708}, right_joints={1: -1.5708})
    report(model, data, "j2 = ±1.20 mirror (real-data init)",
           left_joints={1: 1.20}, right_joints={1: -1.20})
    report(model, data, "j3 = ±π/2  mirror",
           left_joints={2: 1.5708}, right_joints={2: -1.5708})
    report(model, data, "j2=π/2 + j3=π/2 (left)",
           left_joints={1: 1.5708, 2: 1.5708})
    report(model, data, "j2=π/2 + j3=-π/2 (left)",
           left_joints={1: 1.5708, 2: -1.5708})
    report(model, data, "j1 = ±π/2  mirror",
           left_joints={0: -1.5708}, right_joints={0: 1.5708})
    report(model, data, "j1=π/4 + j2=π/2 (left)",
           left_joints={0: 0.7854, 1: 1.5708})


if __name__ == "__main__":
    main()
