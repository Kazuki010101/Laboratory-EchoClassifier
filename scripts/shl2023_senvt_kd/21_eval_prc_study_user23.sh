#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
DATA_ROOT="${SHL_USER23_ROOT:-$PROJECT_ROOT/dataset/2023_processed_user23/100hz_5.0s_overlap0.0s}"
[[ -f "$DATA_ROOT/Hips_Acc.npy" ]] || { echo "Run 11_prepare_shl2023_user23.sh first"; exit 1; }
declare -a SPECS=(
  "PRC|$STUDENT_DISTILLATION_ROOT/standard_kd/PRC_seed${SEED}/best_checkpoint.pth|prc_kd"
  "APS_PRC|$STUDENT_DISTILLATION_ROOT/prc_condensation/aps_ce_seed${SEED}/best_checkpoint.pth|aps_ce"
  "APS_PRC|$STUDENT_DISTILLATION_ROOT/prc_condensation/aps_kd_seed${SEED}/best_checkpoint.pth|aps_kd"
  "TG_SKIP_PRC|$STUDENT_DISTILLATION_ROOT/prc_condensation/tg_skip_seed${SEED}/best_checkpoint.pth|tg_skip"
  "TGEC_PRC|$STUDENT_DISTILLATION_ROOT/prc_condensation/tgec_seed${SEED}/best_checkpoint.pth|tgec"
)
for spec in "${SPECS[@]}"; do
  IFS='|' read -r model ckpt tag <<< "$spec"
  [[ -f "$ckpt" ]] || { echo "[SKIP missing] $ckpt"; continue; }
  bash "$SCRIPT_DIR/12_eval_checkpoint_user23.sh" "$model" "$ckpt" "$tag"
done

