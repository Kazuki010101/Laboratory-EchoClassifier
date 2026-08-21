#!/bin/bash
set -euo pipefail

source transfer_scripts/config_pamap2_all_models.sh

mkdir -p "$PAMAP2_BASELINE_EXP_ROOT"

echo "===== PAMAP2 baseline from scratch: all models ====="
echo "Output root: $PAMAP2_BASELINE_EXP_ROOT"

for cfg in "${PAMAP2_ALL_MODEL_CONFIGS[@]}"; do
  read -r name student patch reservoir rank keep input_rank <<< "$cfg"

  input_rank="${input_rank:-0}"

  for test_subj in $PAMAP2_TEST_SUBJECTS; do
    val_subj=107

    if [ "$test_subj" = "$val_subj" ]; then
      val_subj=108
    fi

    OUT="$PAMAP2_BASELINE_EXP_ROOT/${name}/test${test_subj}_val${val_subj}"
    mkdir -p "$OUT"

    if [ -f "$OUT/test_best.json" ]; then
      echo "========================================"
      echo "[SKIP] result already exists"
      echo "name=$name"
      echo "student=$student"
      echo "test_subject=$test_subj"
      echo "val_subject=$val_subj"
      echo "output_dir=$OUT"
      echo "========================================"
      continue
    fi

    echo "========================================"
    echo "[PAMAP2 Baseline From Scratch]"
    echo "name=$name"
    echo "student=$student"
    echo "patch_size=$patch"
    echo "reservoir_size=$reservoir"
    echo "reservoir_rank=$rank"
    echo "patch_keep_ratio=$keep"
    echo "input_rank=$input_rank"
    echo "test_subject=$test_subj"
    echo "val_subject=$val_subj"
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
      --output_dir "$OUT" \
      2>&1 | tee "$OUT/train_stdout.log"

  done
done

echo "===== DONE PAMAP2 baseline from scratch ====="