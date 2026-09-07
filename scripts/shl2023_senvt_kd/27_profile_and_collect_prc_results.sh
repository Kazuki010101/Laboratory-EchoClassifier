#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PROFILE_DEVICE_VALUE="${PROFILE_DEVICE:-cuda}"
PROFILE_REPEATS_VALUE="${PROFILE_REPEATS:-200}"
PROFILE_WARMUP_VALUE="${PROFILE_WARMUP:-20}"

python "$SCRIPT_DIR/profile_prc_models_v2.py" \
  --output "$RESULTS_ROOT/prc_efficiency_ps16_k16.json" \
  --device "$PROFILE_DEVICE_VALUE" \
  --input-size "$INPUT_SIZE" --reservoir-size 1000 \
  --patch-size 16 --keep-ratio 0.5 \
  --warmup "$PROFILE_WARMUP_VALUE" --repeats "$PROFILE_REPEATS_VALUE"

python "$SCRIPT_DIR/profile_prc_models_v2.py" \
  --output "$RESULTS_ROOT/prc_efficiency_ps16_k24.json" \
  --device "$PROFILE_DEVICE_VALUE" \
  --input-size "$INPUT_SIZE" --reservoir-size 1000 \
  --patch-size 16 --keep-ratio 0.75 \
  --warmup "$PROFILE_WARMUP_VALUE" --repeats "$PROFILE_REPEATS_VALUE"

python "$SCRIPT_DIR/collect_prc_results.py" \
  --experiment-root "$EXPERIMENT_ROOT" \
  --output-dir "$RESULTS_ROOT"

