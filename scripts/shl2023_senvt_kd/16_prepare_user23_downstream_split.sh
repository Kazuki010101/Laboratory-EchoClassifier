#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
LABEL="$SHL_USER23_ROOT/Hips_Label.npy"
[[ -f "$LABEL" ]] || { echo "[ERROR] Run 11_prepare_shl2023_user23.sh first: $LABEL"; exit 1; }
ARGS=()
[[ "${FORCE:-0}" == 1 ]] && ARGS+=(--force)
python "$SCRIPT_DIR/prepare_user23_downstream_split.py" \
  --label "$LABEL" --output "$USER23_SPLIT" --seed "$SEED" "${ARGS[@]}"

