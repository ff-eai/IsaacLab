"""Compute 41-DOF norm stats JSON for a LeRobot v2 dataset.

Output layout matches the ``customized`` branch of
``lingbot-vla/lingbotvla/data/vla_data/transform.Normalizer`` so it can be
selected via ``LINGBOTVLA_CLI`` override at eval time::

    {
      "norm_stats": {
        "observation.state": {"mean": [...], "std": [...], "min": [...],
                              "max": [...], "q01": [...], "q99": [...],
                              "q02": [...], "q98": [...]},
        "action":            { same keys }
      }
    }

Typical use::

    python build_lerobot_norm_stats.py \\
        --dataset_root /workspace/2T/lerobot_cache/a2_pickplace_isaac_57 \\
        --out /workspace/lingbot-vla/assets/norm_stats/a2_pickplace_isaac_57_41d.json
"""

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument(
        "--keys",
        nargs="+",
        default=["observation.state", "action"],
        help="Which fields to summarise (each must be a 1-D array per row).",
    )
    args = parser.parse_args()

    parquet_glob = os.path.join(args.dataset_root, "data/*/*.parquet")
    files = sorted(glob.glob(parquet_glob))
    if not files:
        raise SystemExit(f"No parquet files at {parquet_glob}")
    print(f"[norm] reading {len(files)} parquet files under {args.dataset_root}")

    buckets: dict[str, list[np.ndarray]] = {k: [] for k in args.keys}
    for fp in files:
        df = pd.read_parquet(fp, columns=args.keys)
        for k in args.keys:
            arr = np.stack(df[k].values).astype(np.float32)
            buckets[k].append(arr)
    print(f"[norm] loaded total frames per key: " +
          ", ".join(f"{k}={sum(b.shape[0] for b in v)}" for k, v in buckets.items()))

    out: dict[str, dict] = {}
    for k, parts in buckets.items():
        x = np.concatenate(parts, axis=0)
        n, d = x.shape
        print(f"[norm] {k}: shape={x.shape}")
        stats = {
            "mean": x.mean(axis=0).tolist(),
            "std": x.std(axis=0).tolist(),
            "min": x.min(axis=0).tolist(),
            "max": x.max(axis=0).tolist(),
            "q01": np.quantile(x, 0.01, axis=0).tolist(),
            "q99": np.quantile(x, 0.99, axis=0).tolist(),
            "q02": np.quantile(x, 0.02, axis=0).tolist(),
            "q98": np.quantile(x, 0.98, axis=0).tolist(),
        }
        out[k] = stats

    payload = {"norm_stats": out}
    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[norm] wrote {args.out}")


if __name__ == "__main__":
    main()
