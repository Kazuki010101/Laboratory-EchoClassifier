#!/bin/bash
set -euo pipefail

source transfer_scripts/config_pamap2_transfer.sh

mkdir -p "$PAMAP2_EXP_ROOT"

echo "===== PAMAP2 transfer fine-tuning: all Patch Reservoir family models ====="
echo "Output root: $PAMAP2_EXP_ROOT"

for cfg in "${PAMAP2_TRANSFER_CONFIGS[@]}"; do
  read -r name student patch reservoir rank keep maybe_input_rank maybe_ckpt <<< "$cfg"

  if [ "$student" = "IFSA_LRGR" ]; then
    input_rank="$maybe_input_rank"
    ckpt="$maybe_ckpt"
  else
    input_rank=0
    ckpt="$maybe_input_rank"
  fi

  if [ ! -f "$ckpt" ]; then
    echo "========================================"
    echo "[SKIP] checkpoint not found"
    echo "name=$name"
    echo "checkpoint=$ckpt"
    echo "========================================"
    continue
  fi

  for test_subj in $PAMAP2_TEST_SUBJECTS; do
    val_subj=107

    if [ "$test_subj" = "$val_subj" ]; then
      val_subj=108
    fi

    OUT="$PAMAP2_EXP_ROOT/${name}/test${test_subj}_val${val_subj}"
    mkdir -p "$OUT"

    echo "========================================"
    echo "[PAMAP2 Transfer Fine-tuning]"
    echo "name=$name"
    echo "student=$student"
    echo "patch_size=$patch"
    echo "reservoir_size=$reservoir"
    echo "reservoir_rank=$rank"
    echo "patch_keep_ratio=$keep"
    echo "input_rank=$input_rank"
    echo "test_subject=$test_subj"
    echo "val_subject=$val_subj"
    echo "transfer_checkpoint=$ckpt"
    echo "output_dir=$OUT"
    echo "========================================"

    python transfer_scripts/train_pamap2_transfer.py \
      --data PAMAP2 \
      --pamap2_root "$PAMAP2_ROOT" \
      --pamap2_drop_subj "$PAMAP2_DROP_SUBJ" \
      --fold_test_subj "$test_subj" \
      --fold_val_subj "$val_subj" \
      --input-size 496 \
      --student "$student" \
      --patch_size "$patch" \
      --reservoir_size "$reservoir" \
      --reservoir_rank "$rank" \
      --patch_keep_ratio "$keep" \
      --input_rank "$input_rank" \
      --transfer-checkpoint "$ckpt" \
      --skip-head \
      --distillation-type none \
      --epochs 100 \
      --batch-size 64 \
      --mixup 0 \
      --cutmix 0 \
      --clip-grad 1.0 \
      --lr 5e-4 \
      --warmup-epochs 2 \
      --device cuda \
      --output_dir "$OUT" \
      2>&1 | tee "$OUT/train_stdout.log"

  done
done

echo "===== DONE PAMAP2 transfer fine-tuning ====="