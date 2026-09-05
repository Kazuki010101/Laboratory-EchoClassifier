#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
MODE="${1:-smoke}"
declare -a SPECS=(
  "APS_PRC|$STUDENT_DISTILLATION_ROOT/prc_condensation/aps_ce_seed${SEED}/best_checkpoint.pth|aps_ce_seed${SEED}"
  "APS_PRC|$STUDENT_DISTILLATION_ROOT/prc_condensation/aps_kd_seed${SEED}/best_checkpoint.pth|aps_kd_seed${SEED}"
  "TG_SKIP_PRC|$STUDENT_DISTILLATION_ROOT/prc_condensation/tg_skip_seed${SEED}/best_checkpoint.pth|tg_skip_seed${SEED}"
  "TGEC_PRC|$STUDENT_DISTILLATION_ROOT/prc_condensation/tgec_seed${SEED}/best_checkpoint.pth|tgec_seed${SEED}"
)
for spec in "${SPECS[@]}"; do
  IFS='|' read -r student source tag <<< "$spec"
  bash "$SCRIPT_DIR/17_finetune_checkpoint_user23.sh" "$MODE" "$student" "$source" "$tag"
done

