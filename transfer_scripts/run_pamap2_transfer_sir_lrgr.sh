#!/bin/bash
set -euo pipefail

source transfer_scripts/config_pamap2_all_models.sh

PAMAP2_SIR_TRANSFER_EXP_ROOT="$PAMAP2_TRANSFER_EXP_ROOT"

mkdir -p "$PAMAP2_SIR_TRANSFER_EXP_ROOT"

for cfg in "${PAMAP2_SIR_MODEL_CONFIGS[@]}"; do
  read -r \
    name \
    student \
    patch_size \
    reservoir_size \
    reservoir_rank \
    target_ratio \
    threshold <<< "$cfg"

  transfer_checkpoint="$(
      get_senvtb_transfer_checkpoint "$name"
    )"

  if [ ! -f "$transfer_checkpoint" ]; then
    echo "[ERROR] Transfer checkpoint not found:"
    echo "$transfer_checkpoint"
    exit 1
  fi

  for test_subject in $PAMAP2_TEST_SUBJECTS; do
    if [ "$test_subject" -eq 107 ]; then
      val_subject=108
    else
      val_subject=107
    fi

    output_dir="$PAMAP2_SIR_TRANSFER_EXP_ROOT/${name}/test_subj_${test_subject}"

    mkdir -p "$output_dir"

    echo "=================================================="
    echo "PAMAP2 SIR-LRGR Transfer"
    echo "name=$name"
    echo "student=$student"
    echo "patch_size=$patch_size"
    echo "reservoir_size=$reservoir_size"
    echo "reservoir_rank=$reservoir_rank"
    echo "target_ratio=$target_ratio"
    echo "threshold=$threshold"
    echo "test_subject=$test_subject"
    echo "val_subject=$val_subject"
    echo "checkpoint=$transfer_checkpoint"
    echo "output_dir=$output_dir"
    echo "=================================================="

    python transfer_scripts/train_pamap2_transfer.py \
      $PAMAP2_COMMON_ARGS \
      --student "$student" \
      --patch_size "$patch_size" \
      --reservoir_size "$reservoir_size" \
      --reservoir_rank "$reservoir_rank" \
      --innovation-target-ratio "$target_ratio" \
      --innovation-threshold "$threshold" \
      --innovation-budget-weight 0.1 \
      --innovation-hidden-dim 64 \
      --innovation-min-keep 1 \
      --fold_test_subj "$test_subject" \
      --fold_val_subj "$val_subject" \
      --transfer-checkpoint "$transfer_checkpoint" \
      --skip-head \
      --output_dir "$output_dir" \
      2>&1 | tee "$output_dir/train_stdout.log"

  done
done

echo "=================================================="
echo "PAMAP2 SIR-LRGR transfer completed."
echo "Results: $PAMAP2_SIR_TRANSFER_EXP_ROOT"
echo "=================================================="