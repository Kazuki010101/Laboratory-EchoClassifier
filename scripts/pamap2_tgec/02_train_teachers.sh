#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

MODE="${1:-smoke}"
REQUESTED_FOLD="${2:-all}"
case "$MODE" in
  smoke) EPOCHS=1; WARMUP=0; SUFFIX="_smoke" ;;
  full) EPOCHS="$TEACHER_EPOCHS"; WARMUP="$TEACHER_WARMUP_EPOCHS"; SUFFIX="" ;;
  *) echo "Usage: bash $0 smoke|full [all|101..108]" >&2; exit 2 ;;
esac

require_file "$SPLIT_FILE"
require_file "$SENVT_B_PRETRAINED"
SELECTED_FOLDS="$(select_folds "$REQUESTED_FOLD")"
cd "$PROJECT_ROOT"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

for fold in $SELECTED_FOLDS; do
  out="$TEACHER_ROOT/test${fold}${SUFFIX}"
  if [[ -f "$out/summary.json" && "${FORCE:-0}" != 1 ]]; then
    echo "[SKIP complete] $out"
    continue
  fi
  resume=()
  if [[ -f "$out/checkpoint.pth" && "${FORCE:-0}" != 1 ]]; then
    [[ "${RESUME_PARTIAL:-0}" == 1 ]] || {
      echo "[STOP] Partial run: $out (set RESUME_PARTIAL=1)" >&2; exit 1;
    }
    resume=(--resume "$out/checkpoint.pth")
  fi
  mkdir -p "$out"
  echo "=== PAMAP2 teacher: test_subject=$fold mode=$MODE ==="
  python main.py \
    --data PAMAP2_train --pamap2-root "$PAMAP2_ROOT" \
    --pamap2-drop-subj "$DROP_SUBJECT" --split-indices "$SPLIT_FILE" \
    --fold-test-subj "$fold" --input-size "$INPUT_SIZE" \
    --epochs "$EPOCHS" --batch-size "$TEACHER_BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" --device "$DEVICE" --seed "$TEACHER_SEED" \
    --student senvt-B --pretrained-student-path "$SENVT_B_PRETRAINED" \
    --distillation-type none --mixup 0 --cutmix 0 --smoothing 0.1 \
    --clip-grad 1 --opt adamw --weight-decay "$WEIGHT_DECAY" \
    --lr "$TEACHER_LR" --min-lr "$TEACHER_MIN_LR" \
    --warmup-epochs "$WARMUP" --no-model-ema --unscale-lr \
    --output_dir "$out" "${resume[@]}" 2>&1 | tee "$out/train_stdout.log"
done

