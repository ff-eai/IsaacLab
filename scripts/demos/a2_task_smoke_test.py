"""Smoke test for the A2 place-can-into-tray task.

Checks:
  1. Task modules import (retargeter, env cfg, mimic env cfg)
  2. Gym registry contains Isaac-PickPlace-A2-Abs-v0 + -Mimic-v0
  3. Env cfg class instantiates (no physics yet)
  4. gym.make() creates the env (full init, including USD load + physics)
  5. env.reset() + env.step(idle_action) works end-to-end
"""

from __future__ import annotations

import argparse
import sys
import traceback

# IMPORTANT: pre-import pinocchio before AppLauncher starts. Pinocchio's C++ type
# casters (notably StdVec_StdString) only register in the pybind11 instance active
# at first import. If Isaac Sim's AppLauncher runs first, a subsequent pinocchio
# access (`model.names`) throws:
#     TypeError: No Python class registered for C++ class std::vector<std::string>
# See feedback_a2_physx.md for details.
import pinocchio as _pin_preload  # noqa: F401
from pinocchio.robot_wrapper import RobotWrapper as _RW_preload  # noqa: F401

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app


def step(n: int, msg: str) -> None:
    print(f"\n{'=' * 60}\n[{n}] {msg}\n{'=' * 60}")


failures: list[str] = []


def check(name: str, fn):
    try:
        fn()
        print(f"  ✓ {name}")
    except Exception as e:  # noqa: BLE001
        print(f"  ✗ {name}: {type(e).__name__}: {e}")
        traceback.print_exc()
        failures.append(name)


# --- 1. Import modules --------------------------------------------------------
step(1, "Importing task modules")
check("import A2Retargeter", lambda: __import__(
    "isaaclab.devices.openxr.retargeters.humanoid.agibot.a2_retargeter", fromlist=["A2Retargeter"]))
check("import A2DexRetargeting", lambda: __import__(
    "isaaclab.devices.openxr.retargeters.humanoid.agibot.a2_dex_retargeting_utils",
    fromlist=["A2DexRetargeting"]))
check("import PickPlaceA2EnvCfg", lambda: __import__(
    "isaaclab_tasks.manager_based.manipulation.pick_place.pickplace_a2_env_cfg",
    fromlist=["PickPlaceA2EnvCfg"]))
check("import PickPlaceA2MimicEnvCfg", lambda: __import__(
    "isaaclab_mimic.envs.pinocchio_envs.pickplace_a2_mimic_env_cfg",
    fromlist=["PickPlaceA2MimicEnvCfg"]))
check("import PickPlaceA2MimicEnv", lambda: __import__(
    "isaaclab_mimic.envs.pinocchio_envs.pickplace_a2_mimic_env",
    fromlist=["PickPlaceA2MimicEnv"]))

# --- 2. Gym registry ----------------------------------------------------------
step(2, "Checking gym registry")
import gymnasium as gym
import isaaclab_tasks  # noqa: F401 (triggers task registration)

registered = list(gym.envs.registry.keys())
check("Isaac-PickPlace-A2-Abs-v0 registered",
      lambda: (assert_ := "Isaac-PickPlace-A2-Abs-v0" in registered) or (_ for _ in ()).throw(
          AssertionError(f"not in {[k for k in registered if 'A2' in k]}")))
check("Isaac-PickPlace-A2-Mimic-v0 registered",
      lambda: (assert_ := "Isaac-PickPlace-A2-Mimic-v0" in registered) or (_ for _ in ()).throw(
          AssertionError("not registered")))

# --- 3. Instantiate env cfg ---------------------------------------------------
step(3, "Instantiating env cfg (no physics)")
cfg_ok = [None]


def _make_cfg():
    from isaaclab_tasks.manager_based.manipulation.pick_place.pickplace_a2_env_cfg import PickPlaceA2EnvCfg
    cfg = PickPlaceA2EnvCfg()
    print(f"    decimation={cfg.decimation} dt={cfg.sim.dt} episode_len={cfg.episode_length_s}s")
    print(f"    action_dim={len(cfg.idle_action)}")
    print(f"    scene.num_envs={cfg.scene.num_envs}")
    cfg_ok[0] = cfg


check("PickPlaceA2EnvCfg()", _make_cfg)

# --- 4. gym.make() — full init ------------------------------------------------
step(4, "gym.make('Isaac-PickPlace-A2-Abs-v0') — loads USD + inits physics")
env = [None]


def _make_env():
    # Isaac Lab requires an explicit cfg object — env_cfg_entry_point alone isn't resolved.
    env[0] = gym.make("Isaac-PickPlace-A2-Abs-v0", cfg=cfg_ok[0])
    print(f"    env created: {env[0]}")
    print(f"    obs_space: {env[0].observation_space}")
    print(f"    action_space: {env[0].action_space}")


check("gym.make", _make_env)

# --- 5. reset + step ----------------------------------------------------------
if env[0] is not None:
    step(5, "env.reset() + env.step(idle_action)")

    def _reset_and_step():
        obs, info = env[0].reset()
        print(f"    reset OK")
        import torch
        idle = cfg_ok[0].idle_action.unsqueeze(0)
        obs, rew, term, trunc, info = env[0].step(idle)
        print(f"    step OK — term={term.any().item()} trunc={trunc.any().item()}")

    check("reset+step", _reset_and_step)
    try:
        env[0].close()
    except Exception:  # noqa: BLE001
        pass

# --- Summary ------------------------------------------------------------------
step(99, "Result")
if failures:
    print(f"  ✗ {len(failures)} failure(s): {failures}")
    sys.exit(1)
print("  ✓ all smoke tests passed")
app.close()
