"""Dump all RGB (and optional depth) camera streams from a demo HDF5 file as mp4 videos.

Usage:
    python scripts/demos/hdf5_to_videos.py datasets/a2_pickplace.hdf5 \
        --out datasets/a2_pickplace_videos --fps 30 --depth
"""

import argparse
import os
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np


def is_rgb(ds: h5py.Dataset) -> bool:
    return ds.ndim == 4 and ds.shape[-1] == 3 and ds.dtype == np.uint8


def is_depth(ds: h5py.Dataset) -> bool:
    return ds.ndim == 4 and ds.shape[-1] == 1 and ds.dtype.kind == "f"


def depth_to_rgb(depth: np.ndarray) -> np.ndarray:
    d = depth[..., 0]
    finite = np.isfinite(d)
    if not finite.any():
        return np.zeros((*d.shape, 3), dtype=np.uint8)
    lo, hi = np.percentile(d[finite], [2, 98])
    if hi <= lo:
        hi = lo + 1e-6
    norm = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
    norm = np.where(finite, norm, 0.0)
    g = (255.0 * (1.0 - norm)).astype(np.uint8)
    return np.stack([g, g, g], axis=-1)


def write_video(frames_iter, path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        str(path),
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    ) as w:
        for frame in frames_iter:
            w.append_data(frame)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("hdf5", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--depth", action="store_true", help="also export depth streams")
    ap.add_argument("--demos", nargs="*", default=None, help="restrict to listed demos")
    args = ap.parse_args()

    out_root = args.out or args.hdf5.with_suffix("")
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.hdf5, "r") as f:
        data = f["data"]
        demos = args.demos or list(data.keys())
        for demo in demos:
            obs = data[demo]["obs"]
            for key in obs.keys():
                ds = obs[key]
                if is_rgb(ds):
                    out = out_root / demo / f"{key}.mp4"
                    print(f"[rgb]   {demo}/{key} -> {out}  ({ds.shape})")
                    write_video((ds[i] for i in range(ds.shape[0])), out, args.fps)
                elif args.depth and is_depth(ds):
                    out = out_root / demo / f"{key}.mp4"
                    print(f"[depth] {demo}/{key} -> {out}  ({ds.shape})")
                    write_video(
                        (depth_to_rgb(ds[i]) for i in range(ds.shape[0])),
                        out,
                        args.fps,
                    )

    print(f"Done. Output: {out_root}")


if __name__ == "__main__":
    main()
