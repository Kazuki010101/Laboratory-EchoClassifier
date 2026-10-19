#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

MODE="${1:-smoke}"
case "$MODE" in
  smoke) USER1_EPOCHS=1; USER1_WARMUP=0; FT_MODE=smoke; SUFFIX=_smoke ;;
  full) USER1_EPOCHS="$STUDENT_EPOCHS"; USER1_WARMUP="$STUDENT_WARMUP_EPOCHS"; FT_MODE=full; SUFFIX= ;;
  *) echo "Usage: bash $0 smoke|full" >&2; exit 2 ;;
esac

for f in "$SHL_DISTILL_ACC" "$SHL_DISTILL_LABEL" "$TEACHER_B_CHECKPOINT" \
         "$SHL_USER23_ROOT/Hips_Acc.npy" "$SHL_USER23_ROOT/Hips_Label.npy" "$USER23_SPLIT"; do
  [[ -f "$f" ]] || { echo "Missing input: $f" >&2; exit 1; }
done

cd "$PROJECT_ROOT"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Resume automatically from the last completed epoch when checkpoint.pth exists.
resume_args_for() {
  local out="$1"
  RESUME_ARGS=()
  if [[ -f "$out/checkpoint.pth" ]]; then
    RESUME_ARGS=(--resume "$out/checkpoint.pth")
    echo "[RESUME] $out"
  fi
}

for seed in ${SEEDS:-0 1 2}; do
  [[ "$seed" =~ ^[0-9]+$ ]] || { echo "Invalid seed: $seed" >&2; exit 2; }
  tag="tgec_no_content_seed${seed}"
  user1_out="$STUDENT_DISTILLATION_ROOT/prc_condensation/${tag}${SUFFIX}"
  ft_out="$STUDENT_FINETUNE_ROOT/user23/${tag}${SUFFIX}"
  eval_out="$HELDOUT_EVALUATION_ROOT/user23_test/${tag}${SUFFIX}"

  # Phase A: User1 distillation training
  if [[ ! -f "$user1_out/summary.json" ]]; then
    resume_args_for "$user1_out"
    mkdir -p "$user1_out"
    python main.py --data SHL2023 --input-size "$INPUT_SIZE" \
      --epochs "$USER1_EPOCHS" --batch-size "$STUDENT_BATCH_SIZE" \
      --num_workers "$NUM_WORKERS" --device "$DEVICE" --seed "$seed" \
      --model senvt-B --student TGEC_PRC --teacher-path "$TEACHER_B_CHECKPOINT" \
      --distillation-type soft --distillation-alpha "$DISTILLATION_ALPHA" \
      --distillation-tau "$DISTILLATION_TAU" \
      --patch_size 16 --reservoir_size 1000 --patch_keep_ratio 0.5 \
      --tgec-teacher-layer 1 --tgec-route-weight 1 \
      --tgec-content-weight 0 \
      --mixup 0 --cutmix 0 --smoothing 0.1 \
      --clip-grad 1 --opt adamw --weight-decay "$WEIGHT_DECAY" \
      --lr "$STUDENT_LEARNING_RATE" --min-lr "$MIN_LEARNING_RATE" \
      --warmup-epochs "$USER1_WARMUP" --no-model-ema --unscale-lr \
      --output_dir "$user1_out" "${RESUME_ARGS[@]}" \
      2>&1 | tee -a "$user1_out/train_stdout.log"
  fi
  [[ -f "$user1_out/best_checkpoint.pth" ]] || { echo "Missing User1 best checkpoint" >&2; exit 1; }

  # Phase B: downstream Fine-tuning (reuses existing helper).
  # The helper sources config.sh and sets SEED=0 by default, so for seeds 1/2
  # this helper cannot be called unchanged. Run main.py directly instead.
  if [[ ! -f "$ft_out/summary.json" ]]; then
    resume_args_for "$ft_out"
    mkdir -p "$ft_out"
    if [[ "$MODE" == smoke ]]; then ft_epochs=1; ft_warmup=0;
    else ft_epochs="$DOWNSTREAM_EPOCHS"; ft_warmup="$DOWNSTREAM_WARMUP_EPOCHS"; fi
    python main.py --data SHL2023_user23_finetune \
      --data-path "$SHL_USER23_ROOT" --split-indices "$USER23_SPLIT" \
      --finetune "$user1_out/best_checkpoint.pth" \
      --input-size "$INPUT_SIZE" --epochs "$ft_epochs" \
      --batch-size "$DOWNSTREAM_BATCH_SIZE" --num_workers "$NUM_WORKERS" \
      --device "$DEVICE" --seed "$seed" --student TGEC_PRC \
      --distillation-type none --patch_size 16 --reservoir_size 1000 \
      --reservoir_rank 64 --patch_keep_ratio 0.5 --input_rank 16 \
      --mixup 0 --cutmix 0 --smoothing 0.1 --clip-grad 1 \
      --opt adamw --weight-decay "$WEIGHT_DECAY" \
      --lr "$DOWNSTREAM_LEARNING_RATE" --min-lr "$MIN_LEARNING_RATE" \
      --warmup-epochs "$ft_warmup" --no-model-ema --unscale-lr \
      --output_dir "$ft_out" "${RESUME_ARGS[@]}" \
      2>&1 | tee -a "$ft_out/train_stdout.log"
  fi
  [[ -f "$ft_out/best_checkpoint.pth" ]] || { echo "Missing fine-tuned best checkpoint" >&2; exit 1; }

  # Phase C: held-out User2/3 Test
  if [[ ! -f "$eval_out/evaluation_summary.json" ]]; then
    mkdir -p "$eval_out"
    python main.py --eval --data SHL2023_user23_test \
      --data-path "$SHL_USER23_ROOT" --split-indices "$USER23_SPLIT" \
      --input-size "$INPUT_SIZE" --batch-size "$DOWNSTREAM_BATCH_SIZE" \
      --num_workers "$NUM_WORKERS" --device "$DEVICE" \
      --seed "$seed" --student TGEC_PRC \
      --resume "$ft_out/best_checkpoint.pth" --distillation-type none \
      --patch_size 16 --reservoir_size 1000 --reservoir_rank 64 \
      --patch_keep_ratio 0.5 --input_rank 16 \
      --mixup 0 --cutmix 0 --no-model-ema \
      --output_dir "$eval_out" 2>&1 | tee "$eval_out/eval_stdout.log"
  fi
  echo "[OK] $tag"
done
