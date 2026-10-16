#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

MODE="${1:-smoke}"
TARGET="${2:-all}"
REQUESTED_FOLD="${3:-all}"
case "$MODE" in
  smoke) EPOCHS=1; WARMUP=0; SUFFIX="_smoke" ;;
  full) EPOCHS="$STUDENT_EPOCHS"; WARMUP="$STUDENT_WARMUP_EPOCHS"; SUFFIX="" ;;
  *) echo "Usage: bash $0 smoke|full [all|prc_ce|prc_kd|aps_ce|aps_kd|tg_skip|tgec] [all|101..108]" >&2; exit 2 ;;
esac
case "$TARGET" in
  all) METHODS=(prc_ce prc_kd aps_ce aps_kd tg_skip tgec) ;;
  prc_ce|prc_kd|aps_ce|aps_kd|tg_skip|tgec) METHODS=("$TARGET") ;;
  *) echo "[ERROR] Unknown method: $TARGET" >&2; exit 2 ;;
esac

require_file "$SPLIT_FILE"
SELECTED_FOLDS="$(select_folds "$REQUESTED_FOLD")"
cd "$PROJECT_ROOT"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

for fold in $SELECTED_FOLDS; do
  teacher="$TEACHER_ROOT/test${fold}/best_checkpoint.pth"
  [[ "$MODE" == smoke ]] && teacher="$TEACHER_ROOT/test${fold}_smoke/best_checkpoint.pth"
  for method in "${METHODS[@]}"; do
    case "$method" in
      prc_ce) student=PRC; distill=none; route_w=0; content_w=0 ;;
      prc_kd) student=PRC; distill=soft; route_w=0; content_w=0 ;;
      aps_ce) student=APS_PRC; distill=none; route_w=0; content_w=0 ;;
      aps_kd) student=APS_PRC; distill=soft; route_w=0; content_w=0 ;;
      tg_skip) student=TG_SKIP_PRC; distill=soft; route_w=1; content_w=0 ;;
      tgec) student=TGEC_PRC; distill=soft; route_w=1; content_w=1 ;;
    esac
    [[ "$distill" == none ]] || require_file "$teacher"
    for seed in $SEEDS; do
      out="$STUDENT_ROOT/test${fold}/${method}_seed${seed}${SUFFIX}"
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
      echo "=== PAMAP2 student: fold=$fold method=$method seed=$seed mode=$MODE ==="
      cmd=(python main.py
        --data PAMAP2_train --pamap2-root "$PAMAP2_ROOT"
        --pamap2-drop-subj "$DROP_SUBJECT" --split-indices "$SPLIT_FILE"
        --fold-test-subj "$fold" --input-size "$INPUT_SIZE"
        --epochs "$EPOCHS" --batch-size "$STUDENT_BATCH_SIZE"
        --num_workers "$NUM_WORKERS" --device "$DEVICE" --seed "$seed"
        --student "$student" --model senvt-B --distillation-type "$distill"
        --distillation-alpha "$DISTILLATION_ALPHA" --distillation-tau "$DISTILLATION_TAU"
        --patch_size "$PATCH_SIZE" --reservoir_size "$RESERVOIR_SIZE"
        --patch_keep_ratio "$PATCH_KEEP_RATIO" --tgec-teacher-layer "$TGEC_TEACHER_LAYER"
        --tgec-route-weight "$route_w" --tgec-content-weight "$content_w"
        --mixup 0 --cutmix 0 --smoothing 0.1 --clip-grad 1
        --opt adamw --weight-decay "$WEIGHT_DECAY" --lr "$STUDENT_LR"
        --min-lr "$STUDENT_MIN_LR" --warmup-epochs "$WARMUP"
        --no-model-ema --unscale-lr --output_dir "$out")
      [[ "$distill" == none ]] || cmd+=(--teacher-path "$teacher")
      cmd+=("${resume[@]}")
      "${cmd[@]}" 2>&1 | tee "$out/train_stdout.log"
    done
  done
done

