#!/usr/bin/env bash
# Generate pi0.7-style A2 place-can dataset with Isaac Mimic, then convert to LeRobot.
#
# Stage 1 (Isaac Sim): replay-annotate if needed, mimic-generate synthetic demos.
# Stage 2 (CPU): convert HDF5 -> LeRobot with task/subtask language, subgoal images,
#                 quality/speed metadata, and control-modality labels.
#
# Usage:
#   ./run_a2_pi07_datagen.sh generate   # mimic generation only
#   ./run_a2_pi07_datagen.sh convert    # lerobot conversion only
#   ./run_a2_pi07_datagen.sh all        # both stages
#
# Environment variables (optional overrides):
#   SOURCE_HDF5   - annotated source demos (default: v2 annotated file)
#   OUTPUT_HDF5   - mimic-generated output
#   NUM_TRIALS    - number of mimic generation attempts (default: 500)
#   NUM_ENVS      - parallel envs for generation (default: 4)
#   LEROBOT_HOME  - lerobot dataset cache directory

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ISAAC_PYTHON="${ISAACLAB_ROOT}/_isaac_sim/python.sh"

SOURCE_HDF5="${SOURCE_HDF5:-/home/wagner/code/ext/wagner/dataset/issac_placn/a2_pickplace_v2_annotated.hdf5}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/wagner/code/ext/wagner/dataset/issac_placn/generated}"
OUTPUT_HDF5="${OUTPUT_HDF5:-${OUTPUT_DIR}/a2_pickplace_pi07_generated.hdf5}"
LEROBOT_HOME="${LEROBOT_HOME:-/home/wagner/code/ext/wagner/dataset/issac_placn/lerobot}"
REPO_ID="${REPO_ID:-a2_pickplace_pi07_v2}"
# Python for LeRobot conversion (needs cv2 + lerobot). Override if your venv differs.
CONVERT_PYTHON="${CONVERT_PYTHON:-/home/wagner/code/RoboTwin/.venv/bin/python}"
NUM_TRIALS="${NUM_TRIALS:-500}"
NUM_ENVS="${NUM_ENVS:-4}"
TASK="${TASK:-Isaac-PickPlace-A2-Mimic-v0}"

run_generate() {
  mkdir -p "${OUTPUT_DIR}"
  echo "[pi07-datagen] source=${SOURCE_HDF5}"
  echo "[pi07-datagen] output=${OUTPUT_HDF5} trials=${NUM_TRIALS} envs=${NUM_ENVS}"
  "${ISAAC_PYTHON}" "${SCRIPT_DIR}/isaaclab_mimic/generate_dataset.py" \
    --task "${TASK}" \
    --input_file "${SOURCE_HDF5}" \
    --output_file "${OUTPUT_HDF5}" \
    --num_envs "${NUM_ENVS}" \
    --generation_num_trials "${NUM_TRIALS}" \
    --enable_pinocchio \
    --headless
}

run_convert() {
  local input="${1:-${OUTPUT_HDF5}}"
  echo "[pi07-datagen] converting ${input} -> ${LEROBOT_HOME}/${REPO_ID}"
  "${CONVERT_PYTHON}" "${SCRIPT_DIR}/convert_annotated_to_pi07_lerobot.py" \
    --input_file "${input}" \
    --repo_id "${REPO_ID}" \
    --lerobot_home "${LEROBOT_HOME}" \
    --group auto \
    --control_modality eef
}

stage="${1:-all}"
case "${stage}" in
  generate) run_generate ;;
  convert)  run_convert "${2:-${OUTPUT_HDF5}}" ;;
  convert-source)
    run_convert "${SOURCE_HDF5}"
    ;;
  all)
    run_generate
    run_convert "${OUTPUT_HDF5}"
    ;;
  *)
    echo "Usage: $0 {generate|convert|convert-source|all}" >&2
    exit 1
    ;;
esac
