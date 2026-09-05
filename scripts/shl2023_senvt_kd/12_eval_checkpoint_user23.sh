#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

if [[ $# -lt 3 ]]; then
  echo "Usage: bash $0 STUDENT CHECKPOINT TAG"
  echo "Example: bash $0 PRC path/to/best_checkpoint.pth PRC_standard_kd_seed0"
  exit 1
fi

STUDENT="$1"
CHECKPOINT="$2"
TAG="$3"
DATA_ROOT="${SHL_USER23_ROOT:-$PROJECT_ROOT/dataset/2023_processed_user23/100hz_5.0s_overlap0.0s}"
OUTPUT_DIR="$HELDOUT_EVALUATION_ROOT/user23_mixed/$TAG"
LOG_FILE="$OUTPUT_DIR/eval_stdout.log"
MARKER="$OUTPUT_DIR/completed.txt"

for required in "$CHECKPOINT" "$DATA_ROOT/Hips_Acc.npy" "$DATA_ROOT/Hips_Label.npy"; do
  if [[ ! -f "$required" ]]; then
    echo "[ERROR] Missing required file: $required"
    exit 1
  fi
done

if [[ -f "$MARKER" && "${FORCE:-0}" != "1" ]]; then
  echo "[SKIP] Evaluation already completed: $TAG"
  echo "       $LOG_FILE"
  exit 0
fi

if ! python main.py -h 2>&1 | grep -q 'SHL2023_user23'; then
  echo "[STOP] main.py does not yet accept --data SHL2023_user23."
  echo "Apply the datasets.py/main.py changes described in USER23_DATASET_PATCH.md."
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_ROOT"

echo "=================================================="
echo "Cross-user evaluation: official User2/User3 mixture"
echo "=================================================="
echo "student:     $STUDENT"
echo "checkpoint:  $CHECKPOINT"
echo "data_root:   $DATA_ROOT"
echo "output:      $OUTPUT_DIR"
echo "=================================================="

set +e
python main.py \
  --eval \
  --data SHL2023_user23 \
  --data-path "$DATA_ROOT" \
  --input-size "$INPUT_SIZE" \
  --batch-size "$STUDENT_BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --student "$STUDENT" \
  --resume "$CHECKPOINT" \
  --distillation-type none \
  --patch_size 16 \
  --reservoir_size 1000 \
  --reservoir_rank 64 \
  --patch_keep_ratio 0.5 \
  --input_rank 16 \
  --mixup 0 \
  --cutmix 0 \
  --no-model-ema \
  --output_dir "$OUTPUT_DIR" \
  2>&1 | tee "$LOG_FILE"
EXIT_CODE="${PIPESTATUS[0]}"
set -e

if [[ "$EXIT_CODE" -ne 0 ]]; then
  echo "[ERROR] Evaluation failed with exit code $EXIT_CODE"
  exit "$EXIT_CODE"
fi

{
  echo "completed_at=$(date --iso-8601=seconds)"
  echo "student=$STUDENT"
  echo "checkpoint=$CHECKPOINT"
  echo "dataset=SHL2023_user23"
} > "$MARKER"

echo "[OK] Cross-user evaluation completed: $TAG"

