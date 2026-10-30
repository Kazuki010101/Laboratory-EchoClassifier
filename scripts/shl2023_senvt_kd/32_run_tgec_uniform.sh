
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

MODE="${1:-smoke}"

case "$MODE" in
  smoke)
    USER1_EPOCHS=1
    USER1_WARMUP=0
    FT_EPOCHS=1
    FT_WARMUP=0
    SUFFIX="_smoke"
    ;;
  full)
    USER1_EPOCHS="$STUDENT_EPOCHS"
    USER1_WARMUP="$STUDENT_WARMUP_EPOCHS"
    FT_EPOCHS="$DOWNSTREAM_EPOCHS"
    FT_WARMUP="$DOWNSTREAM_WARMUP_EPOCHS"
    SUFFIX=""
    ;;
  *)
    echo "Usage: bash $0 smoke|full" >&2
    exit 2
    ;;
esac

for f in \
  "$SHL_DISTILL_ACC" \
  "$SHL_DISTILL_LABEL" \
  "$TEACHER_B_CHECKPOINT" \
  "$SHL_USER23_ROOT/Hips_Acc.npy" \
  "$SHL_USER23_ROOT/Hips_Label.npy" \
  "$USER23_SPLIT"; do
  [[ -f "$f" ]] || {
    echo "Missing input: $f" >&2
    exit 1
  }
done

cd "$PROJECT_ROOT"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

resume_args_for() {
  local dir="$1"
  RESUME_ARGS=()
  if [[ -f "$dir/checkpoint.pth" ]]; then
    RESUME_ARGS=(--resume "$dir/checkpoint.pth")
    echo "[RESUME] $dir"
  fi
}

for seed in ${SEEDS:-0 1 2}; do
  [[ "$seed" =~ ^[0-9]+$ ]] || exit 2

  tag="tgec_uniform_seed${seed}"

  user1_out="$STUDENT_DISTILLATION_ROOT/prc_condensation/${tag}${SUFFIX}"
  ft_out="$STUDENT_FINETUNE_ROOT/user23/${tag}${SUFFIX}"
  eval_out="$HELDOUT_EVALUATION_ROOT/user23_test/${tag}${SUFFIX}"

  echo "========== $tag =========="

  # Phase A: User 1 teacher-guided training
  if [[ ! -f "$user1_out/summary.json" ]]; then
    mkdir -p "$user1_out"
    resume_args_for "$user1_out"

    python main.py \
      --data SHL2023 \
      --input-size "$INPUT_SIZE" \
      --epochs "$USER1_EPOCHS" \
      --batch-size "$STUDENT_BATCH_SIZE" \
      --num_workers "$NUM_WORKERS" \
      --device "$DEVICE" \
      --seed "$seed" \
      --model senvt-B \
      --student TGEC_UNIFORM_PRC \
      --teacher-path "$TEACHER_B_CHECKPOINT" \
      --distillation-type soft \
      --distillation-alpha "$DISTILLATION_ALPHA" \
      --distillation-tau "$DISTILLATION_TAU" \
      --patch_size 16 \
      --reservoir_size 1000 \
      --patch_keep_ratio 0.5 \
      --tgec-teacher-layer 1 \
      --tgec-route-weight 0 \
      --tgec-content-weight 1 \
      --mixup 0 --cutmix 0 --smoothing 0.1 \
      --clip-grad 1 \
      --opt adamw \
      --weight-decay "$WEIGHT_DECAY" \
      --lr "$STUDENT_LEARNING_RATE" \
      --min-lr "$MIN_LEARNING_RATE" \
      --warmup-epochs "$USER1_WARMUP" \
      --no-model-ema --unscale-lr \
      --output_dir "$user1_out" \
      "${RESUME_ARGS[@]}" \
      2>&1 | tee -a "$user1_out/train_stdout.log"
  fi

  [[ -f "$user1_out/best_checkpoint.pth" ]] || {
    echo "Missing User1 checkpoint: $tag" >&2
    exit 1
  }

  # Phase B: User 2/3 fine-tuning (CE only)
  if [[ ! -f "$ft_out/summary.json" ]]; then
    mkdir -p "$ft_out"
    resume_args_for "$ft_out"

    python main.py \
      --data SHL2023_user23_finetune \
      --data-path "$SHL_USER23_ROOT" \
      --split-indices "$USER23_SPLIT" \
      --finetune "$user1_out/best_checkpoint.pth" \
      --input-size "$INPUT_SIZE" \
      --epochs "$FT_EPOCHS" \
      --batch-size "$DOWNSTREAM_BATCH_SIZE" \
      --num_workers "$NUM_WORKERS" \
      --device "$DEVICE" \
      --seed "$seed" \
      --student TGEC_UNIFORM_PRC \
      --distillation-type none \
      --patch_size 16 \
      --reservoir_size 1000 \
      --reservoir_rank 64 \
      --patch_keep_ratio 0.5 \
      --input_rank 16 \
      --mixup 0 --cutmix 0 --smoothing 0.1 \
      --clip-grad 1 \
      --opt adamw \
      --weight-decay "$WEIGHT_DECAY" \
      --lr "$DOWNSTREAM_LEARNING_RATE" \
      --min-lr "$MIN_LEARNING_RATE" \
      --warmup-epochs "$FT_WARMUP" \
      --no-model-ema --unscale-lr \
      --output_dir "$ft_out" \
      "${RESUME_ARGS[@]}" \
      2>&1 | tee -a "$ft_out/train_stdout.log"
  fi

  [[ -f "$ft_out/best_checkpoint.pth" ]] || {
    echo "Missing fine-tuned checkpoint: $tag" >&2
    exit 1
  }

  # Phase C: held-out evaluation
  if [[ ! -f "$eval_out/evaluation_summary.json" ]]; then
    mkdir -p "$eval_out"

    python main.py \
      --eval \
      --data SHL2023_user23_test \
      --data-path "$SHL_USER23_ROOT" \
      --split-indices "$USER23_SPLIT" \
      --input-size "$INPUT_SIZE" \
      --batch-size "$DOWNSTREAM_BATCH_SIZE" \
      --num_workers "$NUM_WORKERS" \
      --device "$DEVICE" \
      --seed "$seed" \
      --student TGEC_UNIFORM_PRC \
      --resume "$ft_out/best_checkpoint.pth" \
      --distillation-type none \
      --patch_size 16 \
      --reservoir_size 1000 \
      --reservoir_rank 64 \
      --patch_keep_ratio 0.5 \
      --input_rank 16 \
      --mixup 0 --cutmix 0 --no-model-ema \
      --output_dir "$eval_out" \
      2>&1 | tee "$eval_out/eval_stdout.log"
  fi

  echo "[OK] $tag"
done

echo "[OK] Uniform ablation completed"
