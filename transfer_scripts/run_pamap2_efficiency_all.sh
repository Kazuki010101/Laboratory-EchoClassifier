#!/bin/bash
set -euo pipefail

source transfer_scripts/config_pamap2_all_models.sh

DEVICE="${DEVICE:-cpu}"
OUT_DIR="$EXP_ROOT/pamap2_efficiency"
OUT_CSV="$OUT_DIR/all_models_efficiency.csv"

mkdir -p "$OUT_DIR"

python evaluate_efficiency.py \
  --experiments-root "$PAMAP2_BASELINE_EXP_ROOT" \
  --experiments-root "$PAMAP2_TRANSFER_EXP_ROOT" \
  --pamap2-root "$PAMAP2_ROOT" \
  --device "$DEVICE" \
  --latency-samples 500 \
  --latency-repeats 3 \
  --warmup 50 \
  --output-csv "$OUT_CSV" \
  --overwrite

echo "===== DONE: $OUT_CSV ====="

# cp \
#   "$EXP_ROOT/pamap2_efficiency/all_models_efficiency.csv" \
#   "$EXP_ROOT/pamap2_efficiency/all_models_efficiency_cpu.csv"

# DEVICE=cuda bash transfer_scripts/run_pamap2_efficiency_all.sh

# cp \
#   "$EXP_ROOT/pamap2_efficiency/all_models_efficiency.csv" \
#   "$EXP_ROOT/pamap2_efficiency/all_models_efficiency_gpu.csv"