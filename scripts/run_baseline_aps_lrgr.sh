#!/bin/bash
set -e

source scripts/config_senvt_echo.sh

OUT_ROOT="$EXP_ROOT/stage0_baseline"
mkdir -p "$OUT_ROOT"

echo "[Baseline] Adaptive Patch-Skipping LRGR variants"

for cfg in "${APS_LRGR_CONFIGS[@]}"; do
  read -r name student patch reservoir rank keep <<< "$cfg"

  OUT="$OUT_ROOT/no_kd_to_${name}"
  mkdir -p "$OUT"

  echo "========================================"
  echo "[Baseline] $name"
  echo "student=$student patch=$patch reservoir=$reservoir rank=$rank keep=$keep"
  echo "output_dir=$OUT"
  echo "========================================"

  python main.py $COMMON_ARGS \
    --student "$student" \
    --patch_size "$patch" \
    --reservoir_size "$reservoir" \
    --reservoir_rank "$rank" \
    --patch_keep_ratio "$keep" \
    --distillation-type none \
    --output_dir "$OUT" \
    2>&1 | tee "$OUT/train_stdout.log"
done