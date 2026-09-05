#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

MODE="${1:-smoke}"
case "$MODE" in
  smoke)
    EPOCHS=1
    WARMUP_EPOCHS=0
    SUFFIX="_smoke"
    ;;
  full)
    EPOCHS="$STUDENT_EPOCHS"
    WARMUP_EPOCHS="$STUDENT_WARMUP_EPOCHS"
    SUFFIX=""
    ;;
  *)
    echo "[ERROR] MODE must be smoke or full"
    exit 1
    ;;
esac

OUTPUT_DIR="$STUDENT_BASELINE_ROOT/PRC_ce_seed${SEED}${SUFFIX}"
CHECKPOINT="$OUTPUT_DIR/checkpoint.pth"

if [[ -f "$OUTPUT_DIR/summary.json" && "${FORCE:-0}" != "1" ]]; then
  echo "[SKIP] PRC-CE already completed: $OUTPUT_DIR"
  exit 0
fi

RESUME_ARGS=()
if [[ -f "$CHECKPOINT" && "${RESUME_PARTIAL:-0}" == "1" && "${FORCE:-0}" != "1" ]]; then
  RESUME_ARGS=(--resume "$CHECKPOINT")
elif [[ -f "$CHECKPOINT" && "${FORCE:-0}" != "1" ]]; then
  echo "[STOP] Partial PRC-CE run found."
  echo "Resume: RESUME_PARTIAL=1 bash $0 $MODE"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_ROOT"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

python main.py \
  --data SHL2023 \
  --input-size "$INPUT_SIZE" \
  --epochs "$EPOCHS" \
  --batch-size "$STUDENT_BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --student PRC \
  --distillation-type none \
  --patch_size 16 \
  --reservoir_size 1000 \
  --mixup 0 \
  --cutmix 0 \
  --smoothing 0.1 \
  --clip-grad 1.0 \
  --opt adamw \
  --weight-decay "$WEIGHT_DECAY" \
  --lr "$STUDENT_LEARNING_RATE" \
  --min-lr "$MIN_LEARNING_RATE" \
  --warmup-epochs "$WARMUP_EPOCHS" \
  --no-model-ema \
  --unscale-lr \
  --output_dir "$OUTPUT_DIR" \
  "${RESUME_ARGS[@]}"

