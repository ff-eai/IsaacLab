#!/usr/bin/env bash
# Launch GR00T v5 5cam_abs_ee policy on the training host (172.18.1.26).
# Latest checkpoint: checkpoint-30000 under a2_task6139_v5_5cam_abs_ee.
#
# Usage (from any machine with SSH to 172.18.1.26):
#   bash scripts/imitation_learning/launch_groot_v5_policy_server.sh
#   bash scripts/imitation_learning/launch_groot_v5_policy_server.sh --gpu 2 --port 18007
#
# Eval against it:
#   ./isaaclab.sh -p scripts/imitation_learning/eval_groot_isaac_a2_http_v5_5cam_abs_ee.py \
#     --groot_host 172.18.1.26 --groot_port 18007 --enable_cameras ...

set -euo pipefail

HOST="${GROOT_HOST:-172.18.1.26}"
GPU="${GROOT_GPU:-1}"
PORT="${GROOT_PORT:-18008}"
CKPT="${GROOT_CKPT:-/mnt/extreme_pro/Vincent/GrootN17/checkpoints/a2_task6139_v5_5cam_abs_ee/checkpoint-30000}"
ROOT="/mnt/extreme_pro/Vincent/GrootN17"
LOG="${GROOT_LOG:-/tmp/groot_policy_v5_5cam_abs_ee_${PORT}.log}"

ssh "$HOST" bash -s -- "$GPU" "$PORT" "$CKPT" "$ROOT" "$LOG" <<'REMOTE'
set -euo pipefail
GPU="$1"
PORT="$2"
CKPT="$3"
ROOT="$4"
LOG="$5"

cd "$ROOT"
source env.sh

# Stop prior server on this port (if any).
OLD=$(pgrep -f "policy_server_v5_5cam_abs_ee.py.*--port ${PORT}" || true)
if [[ -n "${OLD}" ]]; then
  echo "Stopping old v5 server PIDs: ${OLD}"
  kill ${OLD} || true
  sleep 2
fi

export CUDA_VISIBLE_DEVICES="${GPU}"
nohup ./.venv/bin/python policy_server_v5_5cam_abs_ee.py \
  --model_path "${CKPT}" \
  --port "${PORT}" \
  --device cuda \
  > "${LOG}" 2>&1 &
echo "Started PID $! on GPU ${GPU} port ${PORT}"
echo "  checkpoint: ${CKPT}"
echo "  log: ${LOG}"

for _ in $(seq 1 120); do
  if grep -qE "policy loaded|Error|Traceback" "${LOG}" 2>/dev/null; then
    tail -20 "${LOG}"
    ss -tlnp 2>/dev/null | grep ":${PORT}" || true
    exit 0
  fi
  sleep 2
done
echo "Timed out waiting for server; tail log:"
tail -30 "${LOG}"
exit 1
REMOTE

echo "v5 policy server launch requested on ${HOST}:${PORT}"
