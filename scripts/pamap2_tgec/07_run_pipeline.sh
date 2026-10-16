#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

STAGE="${1:-smoke}"
case "$STAGE" in
  smoke)
    FOLDS="${FOLDS:-101}" SEEDS="${SEEDS:-0}" bash "$SCRIPT_DIR/01_prepare_splits.sh"
    FOLDS="${FOLDS:-101}" SEEDS="${SEEDS:-0}" bash "$SCRIPT_DIR/02_train_teachers.sh" smoke 101
    FOLDS="${FOLDS:-101}" SEEDS="${SEEDS:-0}" bash "$SCRIPT_DIR/03_run_students.sh" smoke all 101
    FOLDS="${FOLDS:-101}" SEEDS="${SEEDS:-0}" bash "$SCRIPT_DIR/04_evaluate_students.sh" smoke all 101
    ;;
  full)
    bash "$SCRIPT_DIR/01_prepare_splits.sh"
    bash "$SCRIPT_DIR/02_train_teachers.sh" full all
    bash "$SCRIPT_DIR/03_run_students.sh" full all all
    bash "$SCRIPT_DIR/04_evaluate_students.sh" full all all
    bash "$SCRIPT_DIR/05_collect_results.sh"
    bash "$SCRIPT_DIR/06_profile_efficiency.sh"
    ;;
  *) echo "Usage: bash $0 smoke|full" >&2; exit 2 ;;
esac

