#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

CHECKPOINT="$STUDENT_BASELINE_ROOT/PRC_ce_seed${SEED}/best_checkpoint.pth"

bash "$SCRIPT_DIR/12_eval_checkpoint_user23.sh" \
  PRC \
  "$CHECKPOINT" \
  "PRC_ce_seed${SEED}"

