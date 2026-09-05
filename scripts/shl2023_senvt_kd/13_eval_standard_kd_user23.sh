#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

TARGET="${1:-PRC}"

ALL_MODELS=(
  PRC
  PRC_LRGR
  APS_LRGR
  Transformer
  MLPMixer
  Resnet_L
  Resnet_M
  Resnet_S
  DeepConvLSTM100
  DeepConvLSTM50
  DeepConvLSTM25
)

if [[ "$TARGET" == "all_completed" ]]; then
  MODELS=("${ALL_MODELS[@]}")
else
  MODELS=("$TARGET")
fi

for student in "${MODELS[@]}"; do
  run_root="$STUDENT_DISTILLATION_ROOT/standard_kd/${student}_seed${SEED}"
  checkpoint="$run_root/best_checkpoint.pth"

  if [[ ! -f "$checkpoint" ]]; then
    if [[ "$TARGET" == "all_completed" ]]; then
      echo "[SKIP] No completed checkpoint: $student"
      continue
    fi
    echo "[ERROR] Checkpoint not found: $checkpoint"
    exit 1
  fi

  bash "$SCRIPT_DIR/12_eval_checkpoint_user23.sh" \
    "$student" \
    "$checkpoint" \
    "${student}_standard_kd_seed${SEED}"
done

