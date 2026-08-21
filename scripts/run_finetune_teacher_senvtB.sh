#!/bin/bash
set -e

source scripts/config_senvt_echo.sh

OUT="$EXP_ROOT/teacher_finetune/senvtB"
mkdir -p "$OUT"

echo "[Teacher Fine-tune] Pretrained SENvT-B -> SHL2023 classifier"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python main.py --data SHL2023 --input-size 496 \
  --epochs 30 --batch-size 64 \
  --mixup 0 --cutmix 0 --clip-grad 1.0 \
  --student senvt-B \
  --distillation-type none \
  --pretrained-student-path "$PRETRAINED_SENVT_B" \
  --no-model-ema \
  --unscale-lr \
  --lr 1e-5 \
  --min-lr 1e-6 \
  --warmup-epochs 1 \
  --output_dir "$OUT" \
  2>&1 | tee "$OUT/train_stdout.log"