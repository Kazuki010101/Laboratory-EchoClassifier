#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(
  cd "$(dirname "${BASH_SOURCE[0]}")"
  pwd
)"

source "$SCRIPT_DIR/config.sh"

MODE="${1:-smoke}"

case "$MODE" in
  smoke)
    EPOCHS=1
    WARMUP_EPOCHS=0
    OUTPUT_DIR="$TEACHER_ROOT/senvtB_smoke"
    ;;

  full)
    EPOCHS="$TEACHER_EPOCHS"
    WARMUP_EPOCHS="$TEACHER_WARMUP_EPOCHS"
    OUTPUT_DIR="$TEACHER_B_ROOT"
    ;;

  *)
    echo "[ERROR] Unknown mode: $MODE"
    echo
    echo "Usage:"
    echo "  bash $0 smoke"
    echo "  bash $0 full"
    exit 1
    ;;
esac

require_file() {
  local target="$1"

  if [ ! -f "$target" ]; then
    echo "[ERROR] Required file not found:"
    echo "  $target"
    exit 1
  fi
}

require_file "$SHL_DISTILL_ACC"
require_file "$SHL_DISTILL_LABEL"
require_file "$SENVT_B_PRETRAINED"

if [ "$MODE" = "full" ] && \
   [ -f "$OUTPUT_DIR/best_checkpoint.pth" ] && \
   [ "${FORCE:-0}" != "1" ]; then
  echo "[STOP] Full teacher checkpoint already exists:"
  echo "  $OUTPUT_DIR/best_checkpoint.pth"
  echo
  echo "To run again and overwrite it:"
  echo "  FORCE=1 bash $0 full"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

cd "$PROJECT_ROOT"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "=================================================="
echo "SENvT-B fine-tuning on SHL-2023"
echo "=================================================="
echo "mode:                $MODE"
echo "project_root:        $PROJECT_ROOT"
echo "acc:                 $SHL_DISTILL_ACC"
echo "label:               $SHL_DISTILL_LABEL"
echo "pretrained:          $SENVT_B_PRETRAINED"
echo "output:              $OUTPUT_DIR"
echo "epochs:              $EPOCHS"
echo "batch_size:          $BATCH_SIZE"
echo "input_size:          $INPUT_SIZE"
echo "learning_rate:       $LEARNING_RATE"
echo "seed:                $SEED"
echo "=================================================="

  python main.py \
    --data SHL2023 \
    --input-size "$INPUT_SIZE" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --student senvt-B \
    --pretrained-student-path "$SENVT_B_PRETRAINED" \
    --distillation-type none \
    --mixup 0 \
    --cutmix 0 \
    --smoothing 0.1 \
    --clip-grad 1.0 \
    --opt adamw \
    --weight-decay "$WEIGHT_DECAY" \
    --lr "$LEARNING_RATE" \
    --min-lr "$MIN_LEARNING_RATE" \
    --warmup-epochs "$WARMUP_EPOCHS" \
    --no-model-ema \
    --unscale-lr \
    --output_dir "$OUTPUT_DIR"

echo
echo "=================================================="
echo "Teacher fine-tuning completed"
echo "=================================================="
echo "mode:       $MODE"
echo "output:     $OUTPUT_DIR"
echo "checkpoint: $OUTPUT_DIR/best_checkpoint.pth"
echo "summary:    $OUTPUT_DIR/summary.json"
echo "log:        $OUTPUT_DIR/train_stdout.log"
echo "=================================================="