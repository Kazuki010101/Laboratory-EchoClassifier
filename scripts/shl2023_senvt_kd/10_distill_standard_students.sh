#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

MODE="${1:-smoke}"
TARGET="${2:-all}"

case "$MODE" in
  smoke)
    EPOCHS=1
    WARMUP_EPOCHS=0
    RUN_SUFFIX="_smoke"
    ;;
  full)
    EPOCHS="$STUDENT_EPOCHS"
    WARMUP_EPOCHS="$STUDENT_WARMUP_EPOCHS"
    RUN_SUFFIX=""
    ;;
  *)
    echo "[ERROR] MODE must be smoke or full: $MODE"
    exit 1
    ;;
esac

# SIR_LRGR is intentionally excluded.
CORE_MODELS=(
  PRC
  PRC_LRGR
  APS_LRGR
  Transformer
  MLPMixer
)

RESNET_MODELS=(
  Resnet_L
  Resnet_M
  Resnet_S
)

DEEPCONVLSTM_MODELS=(
  DeepConvLSTM100
  DeepConvLSTM50
  DeepConvLSTM25
)

# Optional older reservoir variants; not included in "all".
EXTRA_MODELS=(
  PEAC
  PESAC
  IFSA_LRGR
)

case "$TARGET" in
  all)
    MODELS=(
      "${CORE_MODELS[@]}"
      "${RESNET_MODELS[@]}"
      "${DEEPCONVLSTM_MODELS[@]}"
    )
    ;;
  core)
    MODELS=("${CORE_MODELS[@]}")
    ;;
  resnet)
    MODELS=("${RESNET_MODELS[@]}")
    ;;
  deepconvlstm)
    MODELS=("${DEEPCONVLSTM_MODELS[@]}")
    ;;
  extra)
    MODELS=("${EXTRA_MODELS[@]}")
    ;;
  PRC|PRC_LRGR|APS_LRGR|Transformer|MLPMixer|\
  Resnet_L|Resnet_M|Resnet_S|\
  DeepConvLSTM100|DeepConvLSTM50|DeepConvLSTM25|\
  PEAC|PESAC|IFSA_LRGR)
    MODELS=("$TARGET")
    ;;
  *)
    echo "[ERROR] Unknown target: $TARGET"
    echo "Targets: all, core, resnet, deepconvlstm, extra, or one model name"
    exit 1
    ;;
esac

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "[ERROR] Required file not found: $1"
    exit 1
  fi
}

require_file "$SHL_DISTILL_ACC"
require_file "$SHL_DISTILL_LABEL"
require_file "$TEACHER_B_CHECKPOINT"

# best_checkpoint.pth is produced from the first validation epoch onward.
# summary.json is used as the completion marker so KD cannot start while the
# 100-epoch teacher run is still in progress.
TEACHER_SUMMARY="$TEACHER_B_ROOT/summary.json"
if [[ ! -f "$TEACHER_SUMMARY" ]]; then
  echo "[STOP] Teacher training is not confirmed complete."
  echo "Missing completion marker: $TEACHER_SUMMARY"
  echo "Run this script after the teacher's 100 epochs finish."
  exit 1
fi

cd "$PROJECT_ROOT"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "=================================================="
echo "Standard SENvT-B logit distillation on SHL-2023"
echo "=================================================="
echo "mode:          $MODE"
echo "target:        $TARGET"
echo "models:        ${MODELS[*]}"
echo "teacher:       $TEACHER_B_CHECKPOINT"
echo "epochs:        $EPOCHS"
echo "seed:          $SEED"
echo "alpha:         $DISTILLATION_ALPHA"
echo "temperature:   $DISTILLATION_TAU"
echo "SIR:           excluded"
echo "=================================================="

for STUDENT in "${MODELS[@]}"; do
  OUTPUT_DIR="${STUDENT_DISTILLATION_ROOT}/standard_kd/${STUDENT}_seed${SEED}${RUN_SUFFIX}"
  COMPLETE_MARKER="$OUTPUT_DIR/summary.json"
  LATEST_CHECKPOINT="$OUTPUT_DIR/checkpoint.pth"

  if [[ -f "$COMPLETE_MARKER" && "${FORCE:-0}" != "1" ]]; then
    echo
    echo "[SKIP] Completed: $STUDENT"
    echo "       $COMPLETE_MARKER"
    continue
  fi

  RESUME_ARGS=()
  if [[ -f "$LATEST_CHECKPOINT" && "${FORCE:-0}" != "1" ]]; then
    if [[ "${RESUME_PARTIAL:-0}" == "1" ]]; then
      RESUME_ARGS=(--resume "$LATEST_CHECKPOINT")
      echo "[RESUME] $STUDENT from $LATEST_CHECKPOINT"
    else
      echo
      echo "[STOP] Partial run found for $STUDENT:"
      echo "       $LATEST_CHECKPOINT"
      echo "Resume with: RESUME_PARTIAL=1 bash $0 $MODE $TARGET"
      exit 1
    fi
  fi

  mkdir -p "$OUTPUT_DIR"

  echo
  echo "--------------------------------------------------"
  echo "Student: $STUDENT"
  echo "Output:  $OUTPUT_DIR"
  echo "--------------------------------------------------"

  python main.py \
    --data SHL2023 \
    --input-size "$INPUT_SIZE" \
    --epochs "$EPOCHS" \
    --batch-size "$STUDENT_BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --model senvt-B \
    --student "$STUDENT" \
    --teacher-path "$TEACHER_B_CHECKPOINT" \
    --distillation-type soft \
    --distillation-alpha "$DISTILLATION_ALPHA" \
    --distillation-tau "$DISTILLATION_TAU" \
    --patch_size 16 \
    --reservoir_size 1000 \
    --reservoir_rank 64 \
    --patch_keep_ratio 0.5 \
    --input_rank 16 \
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

  echo "[OK] Completed: $STUDENT"
done

echo
echo "=================================================="
echo "Requested standard-KD runs completed"
echo "=================================================="
