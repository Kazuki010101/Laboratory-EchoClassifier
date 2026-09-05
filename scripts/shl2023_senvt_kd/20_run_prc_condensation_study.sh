#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
MODE="${1:-smoke}"
TARGET="${2:-all}"
case "$MODE" in
  smoke) EPOCHS=1; WARMUP=0; SUFFIX=_smoke ;;
  full) EPOCHS="$STUDENT_EPOCHS"; WARMUP="$STUDENT_WARMUP_EPOCHS"; SUFFIX= ;;
  *) echo "Usage: bash $0 smoke|full [all|aps_ce|aps_kd|tg_skip|tgec]"; exit 2 ;;
esac
case "$TARGET" in
  all) RUNS=(aps_ce aps_kd tg_skip tgec) ;;
  aps_ce|aps_kd|tg_skip|tgec) RUNS=("$TARGET") ;;
  *) echo "Unknown target: $TARGET"; exit 2 ;;
esac
for f in "$SHL_DISTILL_ACC" "$SHL_DISTILL_LABEL" "$TEACHER_B_CHECKPOINT"; do
  [[ -f "$f" ]] || { echo "[ERROR] Missing: $f"; exit 1; }
done
cd "$PROJECT_ROOT"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
for RUN in "${RUNS[@]}"; do
  case "$RUN" in
    aps_ce) STUDENT=APS_PRC; DISTILL=none; ROUTE_W=0; CONTENT_W=0 ;;
    aps_kd) STUDENT=APS_PRC; DISTILL=soft; ROUTE_W=0; CONTENT_W=0 ;;
    tg_skip) STUDENT=TG_SKIP_PRC; DISTILL=soft; ROUTE_W=1; CONTENT_W=0 ;;
    tgec) STUDENT=TGEC_PRC; DISTILL=soft; ROUTE_W=1; CONTENT_W=1 ;;
  esac
  OUT="$STUDENT_DISTILLATION_ROOT/prc_condensation/${RUN}_seed${SEED}${SUFFIX}"
  [[ -f "$OUT/summary.json" && "${FORCE:-0}" != 1 ]] && { echo "[SKIP] $RUN"; continue; }
  RESUME=()
  if [[ -f "$OUT/checkpoint.pth" && "${FORCE:-0}" != 1 ]]; then
    [[ "${RESUME_PARTIAL:-0}" == 1 ]] || { echo "[STOP] Partial: $OUT; use RESUME_PARTIAL=1"; exit 1; }
    RESUME=(--resume "$OUT/checkpoint.pth")
  fi
  mkdir -p "$OUT"
  echo "=== $RUN ($MODE): student=$STUDENT, full-rank PRC, output=$OUT ==="
  python main.py --data SHL2023 --input-size "$INPUT_SIZE" --epochs "$EPOCHS" \
    --batch-size "$STUDENT_BATCH_SIZE" --num_workers "$NUM_WORKERS" --device "$DEVICE" \
    --seed "$SEED" --model senvt-B --student "$STUDENT" --teacher-path "$TEACHER_B_CHECKPOINT" \
    --distillation-type "$DISTILL" --distillation-alpha "$DISTILLATION_ALPHA" \
    --distillation-tau "$DISTILLATION_TAU" --patch_size 16 --reservoir_size 1000 \
    --patch_keep_ratio 0.5 --tgec-teacher-layer 1 --tgec-route-weight "$ROUTE_W" \
    --tgec-content-weight "$CONTENT_W" --mixup 0 --cutmix 0 --smoothing 0.1 \
    --clip-grad 1 --opt adamw --weight-decay "$WEIGHT_DECAY" --lr "$STUDENT_LEARNING_RATE" \
    --min-lr "$MIN_LEARNING_RATE" --warmup-epochs "$WARMUP" --no-model-ema --unscale-lr \
    --output_dir "$OUT" "${RESUME[@]}"
done

