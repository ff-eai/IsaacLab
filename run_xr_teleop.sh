#!/bin/bash
# Launch an Isaac Lab XR teleop task against the running CloudXR Runtime.
# Prerequisite: `./run_cloudxr_runtime.sh` is running in another terminal.
# Usage: ./run_xr_teleop.sh [TASK] [DEVICE]
#   TASK defaults to Isaac-PickPlace-GR1T2-Abs-v0
#   DEVICE defaults to handtracking (or: motion_controllers, manusvive)

# No -u: Isaac Sim's setup_conda_env.sh references $ZSH_VERSION unguarded.
set -eo pipefail

TASK="${1:-Isaac-PickPlace-GR1T2-Abs-v0}"
DEVICE="${2:-handtracking}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Point Isaac Lab at the CloudXR Runtime's OpenXR JSON (shared via bind-mount).
export XDG_RUNTIME_DIR="${SCRIPT_DIR}/openxr/run"
export XR_RUNTIME_JSON="${SCRIPT_DIR}/openxr/share/openxr/1/openxr_cloudxr.json"

# Activate the conda env built during deploy.
source /home/wagner/miniforge3/etc/profile.d/conda.sh
conda activate isaaclab

exec ./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
    --task "$TASK" \
    --teleop_device "$DEVICE" \
    --enable_pinocchio \
    --xr --enable_cameras
