#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
for f in "$SHL_USER23_ROOT/Hips_Acc.npy" "$SHL_USER23_ROOT/Hips_Label.npy" "$USER23_SPLIT"; do
  [[ -f "$f" ]] || { echo "[ERROR] Missing: $f"; exit 1; }
done
declare -a SPECS=(
  "PRC|PRC_ce_seed${SEED}"
  "PRC|PRC_kd_seed${SEED}"
  "APS_PRC|aps_ce_seed${SEED}"
  "APS_PRC|aps_kd_seed${SEED}"
  "TG_SKIP_PRC|tg_skip_seed${SEED}"
  "TGEC_PRC|tgec_seed${SEED}"
)
cd "$PROJECT_ROOT"
for spec in "${SPECS[@]}"; do
  IFS='|' read -r student tag <<< "$spec"
  ckpt="$STUDENT_FINETUNE_ROOT/user23/$tag/best_checkpoint.pth"
  out="$HELDOUT_EVALUATION_ROOT/user23_test/$tag"
  [[ -f "$ckpt" ]] || { echo "[SKIP missing] $ckpt"; continue; }
  if [[ -f "$out/evaluation_summary.json" && "${FORCE:-0}" != 1 ]]; then echo "[SKIP] $tag"; continue; fi
  mkdir -p "$out"
  python main.py --eval --data SHL2023_user23_test --data-path "$SHL_USER23_ROOT" \
    --split-indices "$USER23_SPLIT" --input-size "$INPUT_SIZE" \
    --batch-size "$DOWNSTREAM_BATCH_SIZE" --num_workers "$NUM_WORKERS" --device "$DEVICE" \
    --seed "$SEED" --student "$student" --resume "$ckpt" --distillation-type none \
    --patch_size 16 --reservoir_size 1000 --reservoir_rank 64 --patch_keep_ratio 0.5 \
    --input_rank 16 --mixup 0 --cutmix 0 --no-model-ema --output_dir "$out" \
    2>&1 | tee "$out/eval_stdout.log"
done
