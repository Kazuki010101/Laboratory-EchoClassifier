#!/bin/bash
set -euo pipefail

source transfer_scripts/config_pamap2_all_models.sh

FOOTPRINT_ROOT="$EXP_ROOT/footprint_pamap2_all_models"
FOOTPRINT_CSV="$FOOTPRINT_ROOT/model_static_metrics.csv"

mkdir -p "$FOOTPRINT_ROOT"

echo "===== Footprint: PAMAP2 all models ====="
echo "Output CSV: $FOOTPRINT_CSV"

for cfg in "${PAMAP2_ALL_MODEL_CONFIGS[@]}"; do
  read -r name student patch reservoir rank keep <<< "$cfg"

  echo "========================================"
  echo "[Footprint]"
  echo "name=$name"
  echo "student=$student"
  echo "patch_size=$patch"
  echo "reservoir_size=$reservoir"
  echo "reservoir_rank=$rank"
  echo "patch_keep_ratio=$keep"
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
    --warmup 3 \
    --repeat 20 \
    --csv-path "$FOOTPRINT_CSV"

done

echo "===== DONE Footprint: PAMAP2 all models ====="