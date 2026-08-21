#!/bin/bash
set -e

source scripts/config_senvt_echo.sh

OUT_ROOT="$EXP_ROOT/one_stage_direct/senvtB_to_students"
mkdir -p "$OUT_ROOT"

if [ ! -f "$TEACHER_B" ]; then
  echo "[ERROR] TEACHER_B not found: $TEACHER_B"
  echo "先に bash scripts/run_finetune_teacher_senvtB.sh を実行してください。"
  exit 1
fi

echo "[One-stage] SENvT-B -> IFSA-LRGR variants"

for cfg in "${IFSA_LRGR_CONFIGS[@]}"; do
  read -r name student patch reservoir rank input_rank <<< "$cfg"

  OUT="$OUT_ROOT/senvtB_to_${name}"
  mkdir -p "$OUT"

  echo "========================================"
  echo "[One-stage] SENvT-B -> $name"
  echo "student=$student patch=$patch reservoir=$reservoir rank=$rank input_rank=$input_rank"
  echo "teacher_path=$TEACHER_B"
  echo "output_dir=$OUT"
  echo "========================================"

  python main.py $COMMON_ARGS \
    --student "$student" \
    --patch_size "$patch" \
    --reservoir_size "$reservoir" \
    --reservoir_rank "$rank" \
    --input_rank "$input_rank" \
    --model senvt-B \
    --teacher-path "$TEACHER_B" \
    $KD_ARGS \
    --output_dir "$OUT" \
    2>&1 | tee "$OUT/train_stdout.log"
done