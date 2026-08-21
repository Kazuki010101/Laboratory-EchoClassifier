#!/bin/bash
set -euo pipefail

# source scripts/config_senvt_echo.sh
# source transfer_scripts/config_pamap2_transfer.sh
source transfer_scripts/config_pamap2_all_models.sh

PAMAP2_SIR_BASELINE_EXP_ROOT="$PAMAP2_BASELINE_EXP_ROOT"

mkdir -p "$PAMAP2_SIR_BASELINE_EXP_ROOT"

for cfg in "${PAMAP2_SIR_MODEL_CONFIGS[@]}"; do
  read -r \
    name \
    student \
    patch_size \
    reservoir_size \
    reservoir_rank \
    target_ratio \
    threshold <<< "$cfg"

  for test_subject in $PAMAP2_TEST_SUBJECTS; do
    if [ "$test_subject" -eq 107 ]; then
      val_subject=108
    else
      val_subject=107
    fi

    baseline_name="$name"

    output_dir="$PAMAP2_SIR_BASELINE_EXP_ROOT/${baseline_name}/test_subj_${test_subject}"

    mkdir -p "$output_dir"

    echo "=================================================="
    echo "PAMAP2 SIR-LRGR Baseline"
    echo "name=$baseline_name"
    echo "student=$student"
    echo "patch_size=$patch_size"
    echo "reservoir_size=$reservoir_size"
    echo "reservoir_rank=$reservoir_rank"
    echo "target_ratio=$target_ratio"
    echo "threshold=$threshold"
    echo "test_subject=$test_subject"
    echo "val_subject=$val_subject"
    echo "transfer_checkpoint=NONE"
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
      --transfer-checkpoint "" \
      --output_dir "$output_dir" \
      2>&1 | tee "$output_dir/train_stdout.log"

  done
done

echo "=================================================="
echo "PAMAP2 SIR-LRGR baseline completed."
echo "Results: $PAMAP2_SIR_BASELINE_EXP_ROOT"
echo "=================================================="