#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
cd "$PROJECT_ROOT"
python "$SCRIPT_DIR/profile_prc_models.py" \
  --output "$RESULTS_ROOT/prc_efficiency.json" --device "${PROFILE_DEVICE:-cuda}" \
  --input-size "$INPUT_SIZE" --reservoir-size 1000 --patch-size 16 --keep-ratio 0.5

