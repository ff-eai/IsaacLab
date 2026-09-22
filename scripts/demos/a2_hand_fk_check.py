"""Pure-kinematic feasibility check: can A2's s6_hand enclose a standard can?

Uses pinocchio FK on the A2 URDF directly — no Isaac Sim, no PhysX. Mimic joints
are propagated manually per URDF <mimic> declarations.

For each side, drives the hand to its URDF upper closure limits and reports:
  - thumb-tip → palm distance
  - fingertip → palm distances
  - min thumb↔finger-tip gap (the "enclosure diameter")
  - whether a can of given radius fits between thumb and fingers

Usage:
    python scripts/demos/a2_hand_fk_check.py
    python scripts/demos/a2_hand_fk_check.py --side right --can-radius 0.033
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

try:
    import pinocchio as pin
except ImportError:
    sys.exit("pinocchio is required: pip install pin  (or use the isaaclab conda env)")


URDF_PATH = "/home/wagner/code/IsaacLab/assets/A2/A2.urdf"
MESH_DIR = "/home/wagner/code/IsaacLab/assets/A2"

# Closure targets (rad) = 95% of URDF upper limit per driven joint.
CLOSURE_DRIVEN = {
    "thumb_swing_joint": 2.17,  # swing thumb across palm
    "thumb_1_joint": 0.74,
    "index_1_joint": 1.61,
    "middle_1_joint": 1.61,
    "ring_1_joint": 1.61,
    "pinky_1_joint": 1.61,
}

# Mimic relationships from URDF (per-side): child joint = multiplier * driver.
MIMIC_MAP = {
    "thumb_2_joint": ("thumb_1_joint", 0.40),
    "thumb_3_joint": ("thumb_1_joint", 0.60),
    "index_2_joint": ("index_1_joint", 1.0),
    "middle_2_joint": ("middle_1_joint", 1.0),
    "ring_2_joint": ("ring_1_joint", 1.0),
    "pinky_2_joint": ("pinky_1_joint", 1.0),
}

FINGERTIP_LINKS = ["thumb_3", "index_2", "middle_2", "ring_2", "pinky_2"]


def parse_side(side: str) -> tuple[str, str]:
    """Return (prefix for joints/links like 'L_' or 'R_', palm link name)."""
    if side == "left":
        return "L_", "left_hand"
    return "R_", "right_hand"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=["left", "right"], default="left")
    parser.add_argument("--can-radius", type=float, default=0.0335, help="YCB soup can = 0.0335 m")
    parser.add_argument("--contact-threshold-cm", type=float, default=0.5)
    args = parser.parse_args()

    if not os.path.isfile(URDF_PATH):
        sys.exit(f"URDF not found: {URDF_PATH}")

    print(f"[fk] loading {URDF_PATH} ...")
    model = pin.buildModelFromUrdf(URDF_PATH)
    data = model.createData()
    print(f"[fk] model: nq={model.nq}, njoints={model.njoints}")

    prefix, palm_link = parse_side(args.side)

    # Build full joint value dict from closure targets + mimic propagation.
    driven_full = {f"{prefix}{jn}": v for jn, v in CLOSURE_DRIVEN.items()}
    mimic_full = {
        f"{prefix}{child}": driven_full[f"{prefix}{driver}"] * mult
        for child, (driver, mult) in MIMIC_MAP.items()
    }
    target_by_name = {**driven_full, **mimic_full}

    # Build q vector: zero everywhere except the hand joints we set.
    q = pin.neutral(model)
    set_count = 0
    missing: list[str] = []
    for jname, val in target_by_name.items():
        if not model.existJointName(jname):
            missing.append(jname)
            continue
        jid = model.getJointId(jname)
        # A revolute joint has nq=1 and idx_q points to its slot in q.
        idx_q = model.joints[jid].idx_q
        q[idx_q] = val
        set_count += 1

    print(f"[fk] driven joint values applied:")
    for jn, v in sorted(driven_full.items()):
        print(f"[fk]   {jn:<22s} -> {v:+.3f} rad")
    print(f"[fk] mimic joint values applied:")
    for jn, v in sorted(mimic_full.items()):
        print(f"[fk]   {jn:<22s} -> {v:+.3f} rad")
    if missing:
        print(f"[fk] WARN: {len(missing)} target joints not in model: {missing}")
    print(f"[fk] total joints set: {set_count}/{len(target_by_name)}")

    # Forward kinematics with placements.
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    # Resolve frame IDs by link name.
    def frame_pos(link_name: str) -> np.ndarray:
        if not model.existFrame(link_name):
            sys.exit(f"link '{link_name}' not in pinocchio model frames")
        fid = model.getFrameId(link_name)
        return data.oMf[fid].translation.copy()

    palm_pos = frame_pos(palm_link)
    print(f"\n[fk] palm link '{palm_link}' world pos: {palm_pos.round(4).tolist()}")

    tips = {name: frame_pos(f"{prefix}{name}") for name in FINGERTIP_LINKS}
    print(f"[fk] fingertip world positions:")
    for n, p in tips.items():
        d_palm = np.linalg.norm(p - palm_pos)
        print(f"[fk]   {prefix}{n:<9s}  pos={p.round(4).tolist()}  palm_dist={d_palm*100:.2f} cm")

    thumb = tips["thumb_3"]
    fingers = np.array([tips[k] for k in FINGERTIP_LINKS[1:]])  # index/middle/ring/pinky

    # Minimum distance from thumb tip to each finger tip — the "pinch gap".
    thumb_to_each = np.linalg.norm(fingers - thumb, axis=1)
    print(f"\n[fk] thumb→each finger distances [cm]:")
    for fname, d in zip(FINGERTIP_LINKS[1:], thumb_to_each):
        print(f"[fk]   thumb↔{fname:<9s}: {d*100:6.2f}")

    min_pinch = thumb_to_each.min()
    mean_pinch = thumb_to_each.mean()

    # Simulate placing a can midway between thumb and finger centroid.
    finger_centroid = fingers.mean(axis=0)
    can_center = 0.5 * (thumb + finger_centroid)
    tip_to_can_surface = {n: np.linalg.norm(p - can_center) - args.can_radius for n, p in tips.items()}

    print(f"\n[fk] virtual can center (thumb-finger midpoint): {can_center.round(4).tolist()}")
    print(f"[fk] assumed can radius: {args.can_radius*100:.2f} cm")
    print(f"[fk] fingertip → can surface distances [cm]:")
    for n, d in tip_to_can_surface.items():
        marker = "REACHES" if d * 100 <= args.contact_threshold_cm else "GAP   "
        print(f"[fk]   {prefix}{n:<9s}: {d*100:+7.2f}   {marker}")

    print("\n========== VERDICT ==========")
    print(f"Side:                     {args.side}")
    print(f"Min thumb↔fingertip gap:  {min_pinch*100:.2f} cm   (2 × can_radius = {args.can_radius*200:.2f} cm)")
    print(f"Mean thumb↔fingertip gap: {mean_pinch*100:.2f} cm")

    tip_to_surface = np.array(list(tip_to_can_surface.values()))
    n_reached = int(np.sum(tip_to_surface * 100 <= args.contact_threshold_cm))
    worst_tip = tip_to_surface.max() * 100
    print(f"Fingertips reaching can:  {n_reached} / {len(tip_to_surface)}  (worst {worst_tip:+.2f} cm)")

    # Interpretation.
    if min_pinch < 2 * args.can_radius:
        print("✓ Thumb can close tighter than can diameter — kinematic grasp envelope contains the can.")
    else:
        print(f"✗ Thumb↔fingertip minimum gap ({min_pinch*100:.2f} cm) exceeds can diameter "
              f"({args.can_radius*200:.2f} cm).")
        print("  → Power/wrap grasp requires pushing fingers against can from one side + counter-thumb;")
        print(f"  → Enclosed pinch grasp of this can is NOT feasible with current URDF joint limits.")
        deficit = (min_pinch - 2 * args.can_radius) * 100
        print(f"  → Would need to close an additional ~{deficit:.2f} cm more than URDF upper limits allow.")

    if n_reached == len(tip_to_surface):
        print("✓ All fingertips reach can surface — all 5 digits can contact the can.")
    elif n_reached >= 1:
        print(f"~ {n_reached} of 5 fingertips can reach can surface — partial contact only.")
    else:
        print("✗ No fingertip reaches the can surface with this posture.")


if __name__ == "__main__":
    main()
