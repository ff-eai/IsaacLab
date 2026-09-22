#!/usr/bin/env python3
"""Generate failed episodes until each hand-motion failure category has a minimum count.

Runs Isaac Lab mimic generation in batches, merges new failed episodes into a
master HDF5, and stops when every target category reaches ``min_per_case``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import h5py

SCRIPT_DIR = Path(__file__).resolve().parent
CAPTURE_SCRIPT = SCRIPT_DIR / "capture_failure_dataset.py"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from capture_failure_dataset import classify_hand_motion_result  # noqa: E402

TARGET_CASES = (
    "no_hand_engagement",
    "closed_without_lift",
    "grasp_without_closing",
    "knocked_can_without_grasp",
)

DEFAULT_ISAAC_PYTHON = Path("/home/wagner/code/IsaacLab/_isaac_sim/python.sh")
DEFAULT_GENERATE = Path("/home/wagner/code/IsaacLab/scripts/imitation_learning/isaaclab_mimic/generate_dataset.py")


def count_cases(hdf5_path: Path) -> Counter:
    counts: Counter = Counter()
    with h5py.File(hdf5_path, "r") as f:
        for key in f["data"].keys():
            counts[classify_hand_motion_result(f[f"data/{key}"])] += 1
    return counts


def deficits(counts: Counter, min_per_case: int) -> dict[str, int]:
    return {case: max(0, min_per_case - counts.get(case, 0)) for case in TARGET_CASES}


def all_satisfied(counts: Counter, min_per_case: int) -> bool:
    return all(counts.get(case, 0) >= min_per_case for case in TARGET_CASES)


def next_demo_index(hdf5_path: Path) -> int:
    with h5py.File(hdf5_path, "r") as f:
        ids = [int(k.split("_")[-1]) for k in f["data"].keys()]
    return (max(ids) + 1) if ids else 0


def merge_failed_episodes(src_failed: Path, dst_failed: Path) -> int:
    """Append demos from src into dst with renumbered demo ids. Returns number merged."""
    merged = 0
    with h5py.File(dst_failed, "a") as dst, h5py.File(src_failed, "r") as src:
        next_id = next_demo_index(dst_failed)
        src_keys = sorted(src["data"].keys(), key=lambda k: int(k.split("_")[-1]))
        for src_key in src_keys:
            dst_name = f"demo_{next_id}"
            if dst_name in dst["data"]:
                raise RuntimeError(f"Destination already has {dst_name}")
            src["data"].copy(src_key, dst["data"], dst_name)
            next_id += 1
            merged += 1
        dst["data"].attrs["total"] = len(dst["data"].keys())
    return merged


def run_generation_batch(
    *,
    isaac_python: Path,
    generate_script: Path,
    input_file: Path,
    output_stub: Path,
    batch_trials: int,
    num_envs: int,
    task: str,
    pythonpath: str,
) -> Path:
    output_stub.parent.mkdir(parents=True, exist_ok=True)
    failed_path = Path(str(output_stub) + "_failed.hdf5")
    if failed_path.exists():
        failed_path.unlink()
    stub = Path(str(output_stub) + ".hdf5")
    if stub.exists():
        stub.unlink()

    env = os.environ.copy()
    env["ISAACLAB_PATH"] = "/home/wagner/code/IsaacLab"
    env["PYTHONPATH"] = pythonpath

    cmd = [
        str(isaac_python),
        str(generate_script),
        "--task",
        task,
        "--input_file",
        str(input_file),
        "--output_file",
        str(output_stub) + ".hdf5",
        "--num_envs",
        str(num_envs),
        "--generation_num_trials",
        str(batch_trials),
        "--no-generation_guarantee",
        "--headless",
        # "--enable_cameras",  # BLOCKED: RTX renderer hangs in headless mode on this system (Vulkan+RTX incompatibility)
    ]
    print(f"[balanced_gen] Running batch: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, env=env, check=True)

    if not failed_path.exists():
        raise FileNotFoundError(f"Expected failed output missing: {failed_path}")
    return failed_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate balanced failed-episode dataset.")
    parser.add_argument(
        "--input_file",
        type=Path,
        default=Path("/home/wagner/2T/wagner/dataset/issac_place_can/a2_pickplace_annotated.hdf5"),
    )
    parser.add_argument(
        "--master_failed",
        type=Path,
        default=Path("/home/wagner/2T/wagner/dataset/issac_place_can/failed/a2_pickplace_failed_balanced_failed.hdf5"),
        help="Accumulated failed episodes HDF5",
    )
    parser.add_argument("--min_per_case", type=int, default=25)
    parser.add_argument("--batch_trials", type=int, default=100)
    parser.add_argument("--num_envs", type=int, default=5)
    parser.add_argument("--max_batches", type=int, default=500)
    parser.add_argument("--task", type=str, default="Isaac-PickPlace-A2-Mimic-v0")
    parser.add_argument("--isaac_python", type=Path, default=DEFAULT_ISAAC_PYTHON)
    parser.add_argument("--generate_script", type=Path, default=DEFAULT_GENERATE)
    parser.add_argument(
        "--pythonpath",
        type=str,
        default="/home/wagner/code/IsaacSim/source/python_packages:/home/wagner/code/IsaacLab/source/isaaclab:/home/wagner/code/IsaacLab/source/isaaclab_mimic:/home/wagner/code/IsaacLab/source/isaaclab_tasks",
    )
    parser.add_argument(
        "--seed_from",
        type=Path,
        default=Path("/home/wagner/2T/wagner/dataset/issac_place_can/failed/a2_pickplace_failed_100_failed.hdf5"),
        help="Optional existing failed HDF5 to initialize master",
    )
    args = parser.parse_args()

    master = args.master_failed
    master.parent.mkdir(parents=True, exist_ok=True)

    if not master.exists():
        if args.seed_from.exists():
            print(f"[balanced_gen] Seeding master from {args.seed_from}", flush=True)
            shutil.copy2(args.seed_from, master)
        else:
            raise FileNotFoundError(f"Master missing and seed not found: {master}")

    counts = count_cases(master)
    print(f"[balanced_gen] Initial counts ({sum(counts.values())} episodes): {dict(counts)}", flush=True)
    print(f"[balanced_gen] Deficits (target {args.min_per_case}): {deficits(counts, args.min_per_case)}", flush=True)

    if all_satisfied(counts, args.min_per_case):
        print("[balanced_gen] All categories already satisfied.", flush=True)
        return

    with tempfile.TemporaryDirectory(prefix="balanced_fail_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        for batch_idx in range(1, args.max_batches + 1):
            if all_satisfied(counts, args.min_per_case):
                print(f"[balanced_gen] Done after {batch_idx - 1} extra batches.", flush=True)
                break

            batch_stub = tmpdir_path / f"batch_{batch_idx}"
            batch_failed = run_generation_batch(
                isaac_python=args.isaac_python,
                generate_script=args.generate_script,
                input_file=args.input_file,
                output_stub=batch_stub,
                batch_trials=args.batch_trials,
                num_envs=args.num_envs,
                task=args.task,
                pythonpath=args.pythonpath,
            )

            merged = merge_failed_episodes(batch_failed, master)
            counts = count_cases(master)
            need = deficits(counts, args.min_per_case)
            print(
                f"[balanced_gen] Batch {batch_idx}: merged {merged} episodes -> total {sum(counts.values())}",
                flush=True,
            )
            print(f"[balanced_gen] Counts: {dict(counts)}", flush=True)
            print(f"[balanced_gen] Remaining deficits: {need}", flush=True)
        else:
            print(f"[balanced_gen] Stopped at max_batches={args.max_batches}", flush=True)

    print(f"[balanced_gen] Final master: {master}", flush=True)
    print(f"[balanced_gen] Final counts: {dict(counts)}", flush=True)
    if not all_satisfied(counts, args.min_per_case):
        print(f"[balanced_gen] WARNING: still missing: {deficits(counts, args.min_per_case)}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
