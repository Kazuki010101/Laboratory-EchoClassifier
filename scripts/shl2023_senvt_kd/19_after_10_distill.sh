#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[1/2] Prepare the official User2/User3 mixed validation split"
bash "$SCRIPT_DIR/11_prepare_shl2023_user23.sh"

echo "[2/2] Evaluate the standard-KD PRC checkpoint"
bash "$SCRIPT_DIR/13_eval_standard_kd_user23.sh" PRC

echo "[OK] Immediate post-10_distill stage completed."

