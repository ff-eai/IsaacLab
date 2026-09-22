"""Instantiate the A2 keyboard teleop device through the same factory path that
record_demos.py uses. Reproduces any construction error that would cause
record_demos to exit(1) before the main loop starts.
"""

from __future__ import annotations

import argparse
import sys
import traceback

# Pre-import pinocchio (see feedback_a2_physx.md).
import pinocchio as _pin_preload  # noqa: F401

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

print("[test] stage 1: AppLauncher up")

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab_tasks.manager_based.manipulation.pick_place.pickplace_a2_env_cfg import PickPlaceA2EnvCfg

print("[test] stage 2: modules imported")

cfg = PickPlaceA2EnvCfg()
print(f"[test] stage 3: env cfg ready, teleop devices: {list(cfg.teleop_devices.devices.keys())}")

# Create env (needed because Se3Keyboard subscribes to the app window at init).
env = gym.make("Isaac-PickPlace-A2-Abs-v0", cfg=cfg)
print("[test] stage 4: env created")

try:
    device = create_teleop_device("keyboard", cfg.teleop_devices.devices, callbacks={})
    print(f"[test] stage 5: keyboard device created: {type(device).__name__}")
except Exception as e:
    print(f"[test] stage 5: FAIL {type(e).__name__}: {e}")
    traceback.print_exc()
    env.close()
    app.close()
    sys.exit(1)

try:
    action = device.advance()
    print(f"[test] stage 6: advance() OK — shape={tuple(action.shape)} dtype={action.dtype}")
except Exception as e:
    print(f"[test] stage 6: advance() FAIL {type(e).__name__}: {e}")
    traceback.print_exc()

try:
    # Expand to batch and step the env to check shape compatibility end-to-end.
    actions = action.unsqueeze(0).to(env.unwrapped.device)
    _ = env.step(actions)
    print("[test] stage 7: env.step(keyboard_action) OK")
except Exception as e:
    print(f"[test] stage 7: step FAIL {type(e).__name__}: {e}")
    traceback.print_exc()

env.close()
app.close()
print("[test] done")
