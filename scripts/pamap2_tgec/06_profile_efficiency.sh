#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PROFILE_REPEATS="${PROFILE_REPEATS:-200}"
PROFILE_WARMUP="${PROFILE_WARMUP:-20}"
CPU_THREADS="${CPU_THREADS:-1}"
PROFILER="$PROJECT_ROOT/scripts/shl2023_senvt_kd/profile_prc_models_v2.py"

python "$PROFILER" --output "$RESULTS_ROOT/pamap2_efficiency_gpu.json" \
  --device cuda --input-size "$INPUT_SIZE" --num-classes "$NUM_CLASSES" \
  --reservoir-size "$RESERVOIR_SIZE" --patch-size "$PATCH_SIZE" \
  --keep-ratio "$PATCH_KEEP_RATIO" --batch-size 1 \
  --warmup "$PROFILE_WARMUP" --repeats "$PROFILE_REPEATS"

python "$PROFILER" --output "$RESULTS_ROOT/pamap2_efficiency_cpu.json" \
  --device cpu --input-size "$INPUT_SIZE" --num-classes "$NUM_CLASSES" \
  --reservoir-size "$RESERVOIR_SIZE" --patch-size "$PATCH_SIZE" \
  --keep-ratio "$PATCH_KEEP_RATIO" --batch-size 1 \
  --warmup "$PROFILE_WARMUP" --repeats "$PROFILE_REPEATS" \
  --cpu-threads "$CPU_THREADS"

