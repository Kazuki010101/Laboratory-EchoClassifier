#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

for name in X.npy Y.npy groups.npy; do
  require_file "$PAMAP2_ROOT/$name"
done

cd "$PROJECT_ROOT"
python "$SCRIPT_DIR/prepare_pamap2_loso_splits.py" \
  --root "$PAMAP2_ROOT" \
  --output "$SPLIT_FILE" \
  --manifest "$SPLIT_MANIFEST" \
  --drop-subject "$DROP_SUBJECT" \
  --input-size "$INPUT_SIZE"

