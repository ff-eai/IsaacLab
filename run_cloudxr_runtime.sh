#!/bin/bash
# Start the NVIDIA CloudXR Runtime container for Isaac Lab XR teleop.
# Requires: CloudXR Early Access approval + `docker login nvcr.io`
# The container must be running BEFORE you launch the Isaac Lab teleop script.

set -euo pipefail

IMAGE="${CLOUDXR_IMAGE:-cloudxr-hybrid:5d-6l}"
OPENXR_DIR="$(cd "$(dirname "$0")" && pwd)/openxr"

mkdir -p "$OPENXR_DIR"

# Use -it only when stdin is a TTY; otherwise run detached so the container
# stays alive when invoked from a script or CI.
if [ -t 0 ]; then
    TTY_ARGS="-it"
else
    TTY_ARGS="-d"
fi

exec docker run $TTY_ARGS --rm --name cloudxr-runtime \
    --user "$(id -u):$(id -g)" \
    --gpus=all \
    -e "ACCEPT_EULA=Y" \
    -e "NV_DEVICE_PROFILE=${NV_DEVICE_PROFILE:-auto-webrtc}" \
    -e "NV_CXR_STREAMSDK_ENABLE_ICE=${NV_CXR_STREAMSDK_ENABLE_ICE:-1}" \
    -e "NV_CXR_ENABLE_PUSH_DEVICES=0" \
    --mount "type=bind,src=${OPENXR_DIR},dst=/openxr" \
    --network host \
    "$IMAGE"
