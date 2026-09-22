"""Minimal A2 nomimic spawn: no actuators, no sim.reset() — just load USD and step.

If this hangs, the issue is at USD→articulation physics init, not my actuators.
If it works, the hang is in my A2_NOMIMIC_CFG actuator setup.
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app = AppLauncher(args).app

print("[min] stage 1: after AppLauncher")

import isaaclab.sim as sim_utils

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, device=args.device))
print("[min] stage 2: SimulationContext created")

sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())
sim_utils.DomeLightCfg(intensity=2000.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=2000.0))
print("[min] stage 3: ground + light spawned")

# Spawn USD directly via sim_utils, no Articulation wrapper.
usd_cfg = sim_utils.UsdFileCfg(
    usd_path=os.path.join("/home/wagner/code/IsaacLab/assets/A2", "A2_nomimic.usd"),
)
usd_cfg.func("/World/A2_usd", usd_cfg)
print("[min] stage 4: USD spawned via UsdFileCfg (no Articulation wrapper)")

print("[min] stage 5: calling sim.reset() ...")
sim.reset()
print("[min] stage 6: sim.reset() returned!")

for i in range(10):
    sim.step()
    if i % 2 == 0:
        print(f"[min] step {i} done")
print("[min] SUCCESS")

app.close()
