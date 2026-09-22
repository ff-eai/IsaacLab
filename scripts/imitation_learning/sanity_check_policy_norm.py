"""Sanity-check the deploy-time normalisation of the LingBot WS policy.

No Isaac Sim involved. Loads ``demo_0`` from the source HDF5, sends the
recorded frame-0 observation (state + 3 cameras) to the WS policy server,
and compares the predicted 50-step action chunk against the recorded
``actions[0:50]``. Useful to tell apart "model is just bad at this task" from
"deploy normalisation is mismatched and we're getting garbage actions".

Per-dimension absolute error is printed as a table. If the model is
well-conditioned, predicted ≈ recorded for the early steps, with errors
typically a few cm in position and a few degrees in orientation. If the
deploy norm stats are wrong (e.g. trained un-normalised but eval applies
``bounds_99_woclip``), the predicted chunk will be wildly off, often by
orders of magnitude in some channels.

Usage::

    python sanity_check_policy_norm.py \\
        --dataset_file /home/wagner/2T/wagner/dataset/issac_placn/a2_pickplace_v2_generated.hdf5 \\
        --demo demo_0 --websocket_host 127.0.0.1 --websocket_port 50963
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_file", type=str, required=True)
    parser.add_argument("--demo", type=str, default="demo_0")
    parser.add_argument("--websocket_host", type=str, default="127.0.0.1")
    parser.add_argument("--websocket_port", type=int, default=50963)
    parser.add_argument("--lerobot_path", type=str,
                        default="/home/wagner/code/lingbot-vla")
    parser.add_argument("--task", type=str, default="place the can in the tray")
    parser.add_argument("--n_compare", type=int, default=50,
                        help="How many recorded action rows to compare against the chunk.")
    args = parser.parse_args()

    sys.path.insert(0, args.lerobot_path)
    from deploy.websocket_client_policy import WebsocketClientPolicy

    print(f"[sanity] loading {args.demo} from {args.dataset_file}")
    with h5py.File(args.dataset_file, "r") as f:
        demo = f[f"data/{args.demo}"]
        state_53 = demo["obs/robot_joint_pos"][0].astype(np.float32)
        head = demo["obs/head_camera_rgb"][0]
        cleft = demo["obs/chest_left_camera_rgb"][0]
        cright = demo["obs/chest_right_camera_rgb"][0]
        recorded_actions = demo["actions"][: args.n_compare].astype(np.float32)

    print(f"[sanity] state shape={state_53.shape} cams: head={head.shape} cleft={cleft.shape} cright={cright.shape}")
    print(f"[sanity] recorded actions[0:{args.n_compare}] shape={recorded_actions.shape}")
    print(f"[sanity] recorded action[0] L wrist pos = {recorded_actions[0, 0:3].round(3).tolist()}")
    print(f"[sanity] recorded action[0] R wrist pos = {recorded_actions[0, 7:10].round(3).tolist()}")

    payload = {
        "observation.images.cam_high": np.ascontiguousarray(head),
        "observation.images.cam_left_wrist": np.ascontiguousarray(cleft),
        "observation.images.cam_right_wrist": np.ascontiguousarray(cright),
        "observation.state": state_53,
        "task": args.task,
    }

    print(f"[sanity] connecting ws://{args.websocket_host}:{args.websocket_port}")
    policy = WebsocketClientPolicy(host=args.websocket_host, port=args.websocket_port)
    print(f"[sanity] reset...")
    try:
        policy.reset("isaac_a2")
    except Exception as e:
        print(f"  policy.reset warning: {e}")

    print(f"[sanity] infer...")
    out = policy.infer(payload)
    action = out.get("action")
    if action is None:
        print("  policy returned None — bailing")
        return
    pred = np.asarray(action, dtype=np.float32)
    if pred.ndim == 1:
        pred = pred[None, :]
    print(f"[sanity] predicted chunk shape={pred.shape}")
    print(f"[sanity] predicted action[0] L wrist pos = {pred[0, 0:3].round(3).tolist()}")
    print(f"[sanity] predicted action[0] R wrist pos = {pred[0, 7:10].round(3).tolist()}")

    n = min(pred.shape[0], recorded_actions.shape[0])
    p = pred[:n]
    r = recorded_actions[:n]
    abs_err = np.abs(p - r)
    print(f"\n[sanity] per-dim abs-error over first {n} rows:")
    print(f"{'dim':>4} {'meaning':<20} {'rec_min':>9} {'rec_max':>9} {'pred_min':>9} {'pred_max':>9} {'mae':>8}")

    NAMES = (
        ["L_pos_x", "L_pos_y", "L_pos_z", "L_qw", "L_qx", "L_qy", "L_qz"]
        + ["R_pos_x", "R_pos_y", "R_pos_z", "R_qw", "R_qx", "R_qy", "R_qz"]
        + [f"hand_{i}" for i in range(24)]
    )
    for d in range(38):
        print(
            f"{d:>4} {NAMES[d]:<20} {r[:, d].min():>9.3f} {r[:, d].max():>9.3f} "
            f"{p[:, d].min():>9.3f} {p[:, d].max():>9.3f} {abs_err[:, d].mean():>8.3f}"
        )

    pos_mae = abs_err[:, [0, 1, 2, 7, 8, 9]].mean()
    quat_mae = abs_err[:, [3, 4, 5, 6, 10, 11, 12, 13]].mean()
    hand_mae = abs_err[:, 14:].mean()
    print(f"\n[sanity] summary mae — wrist pos: {pos_mae:.4f}m, quat comp: {quat_mae:.4f}, hand: {hand_mae:.4f}")
    if pos_mae > 0.3:
        print("[sanity] !! wrist position error > 30 cm — strongly suggests bad denorm or wrong frame")
    elif pos_mae > 0.05:
        print("[sanity] wrist position error in 5-30 cm range — typical for an early-trained model")
    else:
        print("[sanity] wrist position error < 5 cm — predictions track recorded actions well")


if __name__ == "__main__":
    main()
