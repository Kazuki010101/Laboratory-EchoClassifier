#!/bin/bash
set -e

source scripts/config_senvt_echo.sh

CSV="$EXP_ROOT/footprint/model_static_metrics.csv"
mkdir -p "$EXP_ROOT/footprint"

echo "[Footprint] IFSA-LRGR variants"

for cfg in "${IFSA_LRGR_CONFIGS[@]}"; do
  read -r name student patch reservoir rank input_rank <<< "$cfg"

  echo "========================================"
  echo "[Footprint] $name"
  echo "student=$student patch=$patch reservoir=$reservoir rank=$rank input_rank=$input_rank"
  echo "========================================"

  python model4edgetest.py \
    --dataset SHL2023 \
    --run-name "$name" \
    --model "$student" \
    --patch_size "$patch" \
    --reservoir_size "$reservoir" \
    --reservoir_rank "$rank" \
    --input_rank "$input_rank" \
    --batch-size 1 \
    --device cpu \
    --csv-path "$CSV"
done