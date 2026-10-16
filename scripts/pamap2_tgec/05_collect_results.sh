#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
cd "$PROJECT_ROOT"

python "$SCRIPT_DIR/aggregate_pamap2_results.py" \
  --evaluation-root "$EVALUATION_ROOT" \
  --output-dir "$RESULTS_ROOT" \
  --expected-folds $FOLDS \
  --expected-seeds $SEEDS

