#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

MODE="${1:-full}"
TARGET="${2:-all}"
REQUESTED_FOLD="${3:-all}"
case "$MODE" in smoke) SUFFIX="_smoke" ;; full) SUFFIX="" ;;
  *) echo "Usage: bash $0 smoke|full [all|method] [all|101..108]" >&2; exit 2 ;;
esac
case "$TARGET" in
  all) METHODS=(prc_ce prc_kd aps_ce aps_kd tg_skip tgec) ;;
  prc_ce|prc_kd|aps_ce|aps_kd|tg_skip|tgec) METHODS=("$TARGET") ;;
  *) echo "[ERROR] Unknown method: $TARGET" >&2; exit 2 ;;
esac

require_file "$SPLIT_FILE"
SELECTED_FOLDS="$(select_folds "$REQUESTED_FOLD")"
cd "$PROJECT_ROOT"

for fold in $SELECTED_FOLDS; do
  for method in "${METHODS[@]}"; do
    case "$method" in
      prc_ce|prc_kd) student=PRC ;;
      aps_ce|aps_kd) student=APS_PRC ;;
      tg_skip) student=TG_SKIP_PRC ;;
      tgec) student=TGEC_PRC ;;
    esac
    for seed in $SEEDS; do
      tag="${method}_seed${seed}${SUFFIX}"
      checkpoint="$STUDENT_ROOT/test${fold}/$tag/best_checkpoint.pth"
      out="$EVALUATION_ROOT/test${fold}/$tag"
      if [[ ! -f "$checkpoint" ]]; then
        echo "[SKIP missing] $checkpoint"
        continue
      fi
      if [[ -f "$out/evaluation_summary.json" && -f "$out/predictions.npz" && "${FORCE:-0}" != 1 ]]; then
        echo "[SKIP complete] $out"
        continue
      fi
      mkdir -p "$out"
      echo "=== PAMAP2 evaluation: fold=$fold method=$method seed=$seed ==="
      python main.py --eval \
        --data PAMAP2_test --pamap2-root "$PAMAP2_ROOT" \
        --pamap2-drop-subj "$DROP_SUBJECT" --split-indices "$SPLIT_FILE" \
        --fold-test-subj "$fold" --input-size "$INPUT_SIZE" \
        --batch-size "$STUDENT_BATCH_SIZE" --num_workers "$NUM_WORKERS" \
        --device "$DEVICE" --seed "$seed" --student "$student" \
        --resume "$checkpoint" --distillation-type none \
        --patch_size "$PATCH_SIZE" --reservoir_size "$RESERVOIR_SIZE" \
        --patch_keep_ratio "$PATCH_KEEP_RATIO" --mixup 0 --cutmix 0 \
        --no-model-ema --output_dir "$out" 2>&1 | tee "$out/eval_stdout.log"
    done
  done
done

