#!/usr/bin/env bash
# Usage: bash 17_finetune_checkpoint_user23.sh smoke|full STUDENT SOURCE_CKPT TAG
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
[[ $# -eq 4 ]] || { echo "Usage: bash $0 smoke|full STUDENT SOURCE_CKPT TAG"; exit 2; }
MODE="$1"; STUDENT="$2"; SOURCE="$3"; TAG="$4"
case "$MODE" in
  smoke) EPOCHS=1; WARMUP=0; SUFFIX=_smoke ;;
  full) EPOCHS="$DOWNSTREAM_EPOCHS"; WARMUP="$DOWNSTREAM_WARMUP_EPOCHS"; SUFFIX= ;;
  *) echo "[ERROR] MODE must be smoke or full"; exit 2 ;;
esac
for f in "$SOURCE" "$SHL_USER23_ROOT/Hips_Acc.npy" "$SHL_USER23_ROOT/Hips_Label.npy" "$USER23_SPLIT"; do
  [[ -f "$f" ]] || { echo "[ERROR] Missing: $f"; exit 1; }
done
OUT="$STUDENT_FINETUNE_ROOT/user23/${TAG}${SUFFIX}"
if [[ -f "$OUT/summary.json" && "${FORCE:-0}" != 1 ]]; then echo "[SKIP] $OUT"; exit 0; fi
RESUME=()
if [[ -f "$OUT/checkpoint.pth" && "${FORCE:-0}" != 1 ]]; then
  [[ "${RESUME_PARTIAL:-0}" == 1 ]] || { echo "[STOP] Partial run: $OUT; use RESUME_PARTIAL=1"; exit 1; }
  RESUME=(--resume "$OUT/checkpoint.pth")
fi
mkdir -p "$OUT"; cd "$PROJECT_ROOT"
echo "=== User2/3 downstream fine-tuning: $TAG ($MODE) ==="
python main.py \
  --data SHL2023_user23_finetune --data-path "$SHL_USER23_ROOT" \
  --split-indices "$USER23_SPLIT" --finetune "$SOURCE" \
  --input-size "$INPUT_SIZE" --epochs "$EPOCHS" --batch-size "$DOWNSTREAM_BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" --device "$DEVICE" --seed "$SEED" --student "$STUDENT" \
  --distillation-type none --patch_size 16 --reservoir_size 1000 --reservoir_rank 64 \
  --patch_keep_ratio 0.5 --input_rank 16 --mixup 0 --cutmix 0 --smoothing 0.1 \
  --clip-grad 1 --opt adamw --weight-decay "$WEIGHT_DECAY" \
  --lr "$DOWNSTREAM_LEARNING_RATE" --min-lr "$MIN_LEARNING_RATE" \
  --warmup-epochs "$WARMUP" --no-model-ema --unscale-lr --output_dir "$OUT" "${RESUME[@]}"

