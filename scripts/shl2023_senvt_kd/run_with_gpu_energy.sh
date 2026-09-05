#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(
  cd "$(dirname "${BASH_SOURCE[0]}")"
  pwd
)"

ENERGY_TOOL="$SCRIPT_DIR/summarize_gpu_energy.py"

if [ "$#" -lt 4 ]; then
  echo "Usage:"
  echo "  $0 OUTPUT_DIR EPOCHS -- COMMAND [ARGS...]"
  exit 1
fi

OUTPUT_DIR="$1"
EPOCHS="$2"
shift 2

if [ "${1:-}" != "--" ]; then
  echo "[ERROR] Missing -- before command."
  echo
  echo "Usage:"
  echo "  $0 OUTPUT_DIR EPOCHS -- COMMAND [ARGS...]"
  exit 1
fi

shift

if [ "$#" -eq 0 ]; then
  echo "[ERROR] Training command is empty."
  exit 1
fi

GPU_INDEX="${GPU_INDEX:-0}"
ENERGY_INTERVAL_MS="${ENERGY_INTERVAL_MS:-500}"

POWER_LOG="$OUTPUT_DIR/gpu_power.csv"
ENERGY_SUMMARY="$OUTPUT_DIR/energy_summary.json"
STDOUT_LOG="$OUTPUT_DIR/train_stdout.log"

mkdir -p "$OUTPUT_DIR"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[ERROR] nvidia-smi was not found."
  exit 1
fi

if [ ! -f "$ENERGY_TOOL" ]; then
  echo "[ERROR] Energy tool was not found:"
  echo "  $ENERGY_TOOL"
  exit 1
fi

if {
  [ -f "$POWER_LOG" ] ||
  [ -f "$ENERGY_SUMMARY" ]
} && [ "${ENERGY_FORCE:-0}" != "1" ]; then
  echo "[STOP] Existing energy log found:"
  echo "  $OUTPUT_DIR"
  echo
  echo "To overwrite:"
  echo "  ENERGY_FORCE=1 ..."
  exit 1
fi

rm -f "$POWER_LOG"
rm -f "$ENERGY_SUMMARY"

MONITOR_PID=""

stop_monitor() {
  if [ -n "$MONITOR_PID" ] && \
     kill -0 "$MONITOR_PID" 2>/dev/null; then
    kill -TERM "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
  fi
}

trap stop_monitor EXIT INT TERM

echo "========================================"
echo "GPU ENERGY MONITOR"
echo "========================================"
echo "GPU index:       $GPU_INDEX"
echo "Interval:        $ENERGY_INTERVAL_MS ms"
echo "Power log:       $POWER_LOG"
echo "Energy summary:  $ENERGY_SUMMARY"
echo "========================================"

python "$ENERGY_TOOL" monitor \
  --output "$POWER_LOG" \
  --gpu-index "$GPU_INDEX" \
  --interval-ms "$ENERGY_INTERVAL_MS" &

MONITOR_PID=$!

sleep 1

if ! kill -0 "$MONITOR_PID" 2>/dev/null; then
  echo "[ERROR] GPU monitor failed to start."
  wait "$MONITOR_PID" || true
  exit 1
fi

START_EPOCH="$(
  python -c 'import time; print(time.time())'
)"

set +e

"$@" 2>&1 | tee "$STDOUT_LOG"
TRAINING_EXIT_CODE="${PIPESTATUS[0]}"

set -e

END_EPOCH="$(
  python -c 'import time; print(time.time())'
)"

stop_monitor
MONITOR_PID=""

trap - EXIT INT TERM

python "$ENERGY_TOOL" summarize \
  --input "$POWER_LOG" \
  --output "$ENERGY_SUMMARY" \
  --start-epoch "$START_EPOCH" \
  --end-epoch "$END_EPOCH" \
  --epochs "$EPOCHS" \
  --exit-code "$TRAINING_EXIT_CODE"

if [ "$TRAINING_EXIT_CODE" -ne 0 ]; then
  echo
  echo "[ERROR] Training command failed."
  echo "Exit code: $TRAINING_EXIT_CODE"
  exit "$TRAINING_EXIT_CODE"
fi

echo
echo "[OK] Training and energy measurement completed."
echo "Training log:  $STDOUT_LOG"
echo "Power log:     $POWER_LOG"
echo "Energy result: $ENERGY_SUMMARY"