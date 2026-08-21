#!/bin/bash
set -e

source scripts/config_senvt_echo.sh

OUT_ROOT="$EXP_ROOT/stage0_baseline"
mkdir -p "$OUT_ROOT"

echo "[Baseline] IFSA-LRGR variants"

for cfg in "${IFSA_LRGR_CONFIGS[@]}"; do
  read -r name student patch reservoir rank input_rank <<< "$cfg"

  OUT="$OUT_ROOT/no_kd_to_${name}"
  mkdir -p "$OUT"

  echo "========================================"
  echo "[Baseline] $name"
  echo "student=$student patch=$patch reservoir=$reservoir rank=$rank input_rank=$input_rank"
  echo "output_dir=$OUT"
  echo "========================================"

  python main.py $COMMON_ARGS \
    --student "$student" \
    --patch_size "$patch" \
    --reservoir_size "$reservoir" \
    --reservoir_rank "$rank" \
    --input_rank "$input_rank" \
    --distillation-type none \
    --output_dir "$OUT" \
    2>&1 | tee "$OUT/train_stdout.log"
done