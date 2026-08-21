#!/bin/bash
set -e

source scripts/config_senvt_echo.sh

OUT_ROOT="$EXP_ROOT/stage1_intermediate_teacher"
mkdir -p "$OUT_ROOT"

echo "[Stage1] SENvT-B -> SENvT-XS"

python main.py $COMMON_ARGS \
  --student senvt-XS \
  --model senvt-B \
  --teacher-path "$TEACHER_B" \
  $KD_ARGS \
  --output_dir "$OUT_ROOT/senvtB_to_senvtXS"

echo "[Stage1] SENvT-B -> SENvT-S"

python main.py $COMMON_ARGS \
  --student senvt-S \
  --model senvt-B \
  --teacher-path "$TEACHER_B" \
  $KD_ARGS \
  --output_dir "$OUT_ROOT/senvtB_to_senvtS"
