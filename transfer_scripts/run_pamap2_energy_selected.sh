#!/bin/bash
set -euo pipefail

source transfer_scripts/config_pamap2_all_models.sh

DEVICE="${DEVICE:-cuda}"
MODEL_FILTER="${MODEL_FILTER:-PRC_p64_r1000,LRGR_p64_r1000_rank32,APS_LRGR_p64_r1000_rank32_keep050,SIR_LRGR_p64_r1000_rank32_target050}"
OUT_DIR="$EXP_ROOT/pamap2_efficiency"
OUT_CSV="$OUT_DIR/selected_models_energy.csv"

mkdir -p "$OUT_DIR"

python evaluate_efficiency.py \
  --experiments-root "$PAMAP2_BASELINE_EXP_ROOT" \
  --experiments-root "$PAMAP2_TRANSFER_EXP_ROOT" \
  --pamap2-root "$PAMAP2_ROOT" \
  --device "$DEVICE" \
  --model-filter "$MODEL_FILTER" \
  --latency-samples 500 \
  --latency-repeats 5 \
  --warmup 100 \
  --measure-energy \
  --energy-seconds 30 \
  --idle-seconds 10 \
  --output-csv "$OUT_CSV" \
  --overwrite

echo "===== DONE: $OUT_CSV ====="
