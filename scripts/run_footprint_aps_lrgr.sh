#!/bin/bash
set -e

source scripts/config_senvt_echo.sh

CSV="$EXP_ROOT/footprint/model_static_metrics.csv"
mkdir -p "$EXP_ROOT/footprint"

echo "[Footprint] Adaptive Patch-Skipping LRGR variants"

for cfg in "${APS_LRGR_CONFIGS[@]}"; do
  read -r name student patch reservoir rank keep <<< "$cfg"

  echo "========================================"
  echo "[Footprint] $name"
  echo "student=$student patch=$patch reservoir=$reservoir rank=$rank keep=$keep"
  echo "========================================"

  python model4edgetest.py \
    --dataset SHL2023 \
    --run-name "$name" \
    --model "$student" \
    --patch_size "$patch" \
    --reservoir_size "$reservoir" \
    --reservoir_rank "$rank" \
    --patch_keep_ratio "$keep" \
    --batch-size 1 \
    --device cpu \
    --csv-path "$CSV"
done