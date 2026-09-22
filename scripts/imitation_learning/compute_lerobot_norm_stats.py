"""Compute per-dim q01 / q99 stats for an existing LeRobot v2 dataset and
write a norm_stats JSON that LingBot-VLA's Normalizer (data_type='customized'
+ norm_type='bounds_99_woclip') can consume.

Run inside the RoboTwin .venv (has lerobot + h5py).

Usage::

  /home/wagner/code/RoboTwin/.venv/bin/python \
      /home/wagner/code/IsaacLab/scripts/imitation_learning/compute_lerobot_norm_stats.py \
      --lerobot_home /home/wagner/2T/wagner/lerobot_cache \
      --repo_id a2_pickplace_v2_39 \
      --out /home/wagner/2T/wagner/norm_stats/a2_pickplace_v2_39.json
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lerobot_home", required=True)
    ap.add_argument("--repo_id", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--keys",
        nargs="+",
        default=["observation.state", "action"],
        help="LeRobot column names to compute q01/q99 over.",
    )
    args = ap.parse_args()

    os.environ["HF_LEROBOT_HOME"] = args.lerobot_home
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    print(f"[norm-stats] loading {args.repo_id} from {args.lerobot_home}")
    ds = LeRobotDataset(repo_id=args.repo_id, root=Path(args.lerobot_home) / args.repo_id)
    print(f"[norm-stats] frames: {len(ds)}")

    column_buf = {k: [] for k in args.keys}
    for i, item in enumerate(ds):
        for k in args.keys:
            v = item[k]
            if hasattr(v, "detach"):
                v = v.detach().cpu().numpy()
            column_buf[k].append(np.asarray(v, dtype=np.float64))
        if i % 1000 == 0:
            print(f"  ... {i}/{len(ds)} frames")

    norm_stats = {}
    DEAD_DIM_RANGE = 1e-2  # range below this → treat as constant, no rescale
    for k, rows in column_buf.items():
        arr = np.stack(rows, axis=0)  # (N, D)
        q01 = np.quantile(arr, 0.01, axis=0)
        q99 = np.quantile(arr, 0.99, axis=0)
        # For dims with effectively-zero range (locked joints, padding, etc.)
        # set q01=-1, q99=1 so bounds_99_woclip becomes
        # `(value - (-1)) / 2 * 2 - 1 ≈ value` for value~0 — i.e. identity, not
        # a divide-by-near-zero blowup.
        dead = (q99 - q01) < DEAD_DIM_RANGE
        q01 = np.where(dead, -1.0, q01)
        q99 = np.where(dead, 1.0, q99)
        n_dead = int(dead.sum())
        norm_stats[k] = {
            "q01": q01.tolist(),
            "q99": q99.tolist(),
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
        }
        print(
            f"  {k}: shape=(N={arr.shape[0]}, D={arr.shape[1]})  dead_dims={n_dead}/{arr.shape[1]} "
            f"(set to q01=-1,q99=1).  active q01[min,max]=[{q01[~dead].min():.3f},{q01[~dead].max():.3f}] "
            f"q99[min,max]=[{q99[~dead].min():.3f},{q99[~dead].max():.3f}]"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "_meta": {
            "name": f"{args.repo_id}_q01q99",
            "source": f"computed from {args.repo_id} ({len(ds)} frames)",
            "layout": "customized",
        },
        "norm_stats": norm_stats,
    }, indent=2))
    print(f"[norm-stats] wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
