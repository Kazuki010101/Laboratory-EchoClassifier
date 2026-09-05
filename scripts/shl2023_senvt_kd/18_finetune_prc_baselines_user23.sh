#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
MODE="${1:-smoke}"
bash "$SCRIPT_DIR/17_finetune_checkpoint_user23.sh" "$MODE" PRC \
  "$STUDENT_BASELINE_ROOT/PRC_ce_seed${SEED}/best_checkpoint.pth" "PRC_ce_seed${SEED}"
bash "$SCRIPT_DIR/17_finetune_checkpoint_user23.sh" "$MODE" PRC \
  "$STUDENT_DISTILLATION_ROOT/standard_kd/PRC_seed${SEED}/best_checkpoint.pth" "PRC_kd_seed${SEED}"

