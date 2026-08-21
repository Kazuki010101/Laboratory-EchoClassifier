#!/usr/bin/env bash
set -euo pipefail

source transfer_scripts/config_pamap2_all_models.sh

mkdir -p "$PAMAP2_TRANSFER_EXP_ROOT"

echo "===== SHL to PAMAP2 transfer fine-tuning: IFSA-LRGR ====="
echo "Output root: $PAMAP2_TRANSFER_EXP_ROOT"

for cfg in "${PAMAP2_IFSA_MODEL_CONFIGS[@]}"; do
  read -r name student patch reservoir rank keep input_rank <<< "$cfg"

  CKPT="$(get_senvtb_transfer_checkpoint "$name")"

  if [ ! -f "$CKPT" ]; then
    echo "========================================"
    echo "[SKIP] SHL distilled checkpoint not found"
    echo "name=$name"
    echo "checkpoint=$CKPT"
    echo "========================================"
    continue
  fi

  for test_subj in $PAMAP2_TEST_SUBJECTS; do
    val_subj=107

    if [ "$test_subj" = "$val_subj" ]; then
      val_subj=108
    fi

    OUT="$PAMAP2_TRANSFER_EXP_ROOT/${name}/test${test_subj}_val${val_subj}"
    mkdir -p "$OUT"

    echo "========================================"
    echo "[SHL to PAMAP2 Transfer Fine-tuning]"
    echo "name=$name"
    echo "student=$student"
    echo "patch_size=$patch"
    echo "reservoir_size=$reservoir"
    echo "reservoir_rank=$rank"
    echo "patch_keep_ratio=$keep"
    echo "input_rank=$input_rank"
    echo "test_subject=$test_subj"
    echo "val_subject=$val_subj"
    echo "transfer_checkpoint=$CKPT"
    echo "output_dir=$OUT"
    echo "========================================"

    python transfer_scripts/train_pamap2_transfer.py \
      $PAMAP2_COMMON_ARGS \
      --fold_test_subj "$test_subj" \
      --fold_val_subj "$val_subj" \
      --student "$student" \
      --patch_size "$patch" \
      --reservoir_size "$reservoir" \
      --reservoir_rank "$rank" \
      --patch_keep_ratio "$keep" \
      --input_rank "$input_rank" \
      --transfer-checkpoint "$CKPT" \
      --skip-head \
      --output_dir "$OUT" \
      2>&1 | tee "$OUT/train_stdout.log"
  done
done

echo "===== DONE SHL to PAMAP2 transfer fine-tuning: IFSA-LRGR ====="