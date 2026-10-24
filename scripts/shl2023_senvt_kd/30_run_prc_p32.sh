# SEEDS="0 1 2" bash scripts/shl2023_senvt_kd/30_run_prc_p32.sh full

# SEEDS="0 1 2" RESUME_PARTIAL=1 \
# bash scripts/shl2023_senvt_kd/30_run_prc_p32.sh full

# find experiments/shl2023_senvt_kd/heldout_evaluation/user23_test \
#   -path '*/prc_p32_*_seed*/evaluation_summary.json' | sort

# # CPU
# python scripts/shl2023_senvt_kd/profile_prc_models_v2.py \
#   --output experiments/shl2023_senvt_kd/results/prc_p32_cpu.json \
#   --device cpu \
#   --patch-size 32 \
#   --warmup 20 \
#   --repeats 200 \
#   --cpu-threads 1

# # GPU
# python scripts/shl2023_senvt_kd/profile_prc_models_v2.py \
#   --output experiments/shl2023_senvt_kd/results/prc_p32_gpu.json \
#   --device cuda \
#   --patch-size 32 \
#   --warmup 20 \
#   --repeats 200

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

# Usage:
# SEEDS="0" bash scripts/shl2023_senvt_kd/30_run_prc_p32.sh smoke
# SEEDS="0 1 2" bash scripts/shl2023_senvt_kd/30_run_prc_p32.sh full

MODE="${1:-smoke}"
SEEDS_VALUE="${SEEDS:-0 1 2}"
METHODS=(ce kd)

PATCH_SIZE=32
RESERVOIR_SIZE=1000

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
    echo "[ERROR] Use smoke or full"
    exit 2
    ;;
esac

for f in \
  "$SHL_DISTILL_ACC" \
  "$SHL_DISTILL_LABEL" \
  "$SHL_USER23_ROOT/Hips_Acc.npy" \
  "$SHL_USER23_ROOT/Hips_Label.npy" \
  "$USER23_SPLIT" \
  "$TEACHER_B_CHECKPOINT"; do
  [[ -f "$f" ]] || {
    echo "[ERROR] Missing: $f"
    exit 1
  }
done

cd "$PROJECT_ROOT"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Resume from an interrupted training run.
resume_args_for() {
  local out="$1"
  RESUME_ARGS=()
  if [[ -f "$out/checkpoint.pth" ]]; then
    if [[ "${RESUME_PARTIAL:-0}" == "1" ]]; then
      RESUME_ARGS=(--resume "$out/checkpoint.pth")
      echo "[RESUME] $out"
    else
      echo "[STOP] Partial checkpoint: $out"
      echo "Use RESUME_PARTIAL=1 to resume."
      exit 1
    fi
  fi
}

run_user1() {
  local seed="$1"
  local method="$2"
  local tag="prc_p32_${method}_seed${seed}"
  local out="$STUDENT_DISTILLATION_ROOT/prc_p32/${tag}${SUFFIX}"

  if [[ -f "$out/summary.json" ]]; then
    echo "[SKIP User1] $tag"
    return
  fi

  mkdir -p "$out"
  resume_args_for "$out"

  local kd_args=()
  if [[ "$method" == "kd" ]]; then
    kd_args=(
      --model senvt-B
      --teacher-path "$TEACHER_B_CHECKPOINT"
      --distillation-type soft
      --distillation-alpha "$DISTILLATION_ALPHA"
      --distillation-tau "$DISTILLATION_TAU"
    )
  else
    kd_args=(--distillation-type none)
  fi

  echo "=== User1: $tag ==="

  python main.py \
    --data SHL2023 \
    --input-size "$INPUT_SIZE" \
    --epochs "$USER1_EPOCHS" \
    --batch-size "$STUDENT_BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --device "$DEVICE" \
    --seed "$seed" \
    --student PRC \
    --patch_size "$PATCH_SIZE" \
    --reservoir_size "$RESERVOIR_SIZE" \
    "${kd_args[@]}" \
    --mixup 0 \
    --cutmix 0 \
    --smoothing 0.1 \
    --clip-grad 1 \
    --opt adamw \
    --weight-decay "$WEIGHT_DECAY" \
    --lr "$STUDENT_LEARNING_RATE" \
    --min-lr "$MIN_LEARNING_RATE" \
    --warmup-epochs "$USER1_WARMUP" \
    --no-model-ema \
    --unscale-lr \
    --output_dir "$out" \
    "${RESUME_ARGS[@]}" \
    2>&1 | tee -a "$out/train_stdout.log"

  [[ -f "$out/best_checkpoint.pth" ]] || {
    echo "[ERROR] Missing User1 best checkpoint: $tag"
    exit 1
  }
}

run_finetune() {
  local seed="$1"
  local method="$2"
  local tag="prc_p32_${method}_seed${seed}"

  local source="$STUDENT_DISTILLATION_ROOT/prc_p32/${tag}${SUFFIX}/best_checkpoint.pth"
  local out="$STUDENT_FINETUNE_ROOT/user23/${tag}${SUFFIX}"

  [[ -f "$source" ]] || {
    echo "[ERROR] Missing source checkpoint: $source"
    exit 1
  }

  if [[ -f "$out/summary.json" ]]; then
    echo "[SKIP FT] $tag"
    return
  fi

  mkdir -p "$out"
  resume_args_for "$out"

  echo "=== User2/User3 Fine-tuning: $tag ==="

  python main.py \
    --data SHL2023_user23_finetune \
    --data-path "$SHL_USER23_ROOT" \
    --split-indices "$USER23_SPLIT" \
    --finetune "$source" \
    --input-size "$INPUT_SIZE" \
    --epochs "$FT_EPOCHS" \
    --batch-size "$DOWNSTREAM_BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --device "$DEVICE" \
    --seed "$seed" \
    --student PRC \
    --distillation-type none \
    --patch_size "$PATCH_SIZE" \
    --reservoir_size "$RESERVOIR_SIZE" \
    --mixup 0 \
    --cutmix 0 \
    --smoothing 0.1 \
    --clip-grad 1 \
    --opt adamw \
    --weight-decay "$WEIGHT_DECAY" \
    --lr "$DOWNSTREAM_LEARNING_RATE" \
    --min-lr "$MIN_LEARNING_RATE" \
    --warmup-epochs "$FT_WARMUP" \
    --no-model-ema \
    --unscale-lr \
    --output_dir "$out" \
    "${RESUME_ARGS[@]}" \
    2>&1 | tee -a "$out/train_stdout.log"

  [[ -f "$out/best_checkpoint.pth" ]] || {
    echo "[ERROR] Missing FT best checkpoint: $tag"
    exit 1
  }
}

run_test() {
  local seed="$1"
  local method="$2"
  local tag="prc_p32_${method}_seed${seed}"

  local ckpt="$STUDENT_FINETUNE_ROOT/user23/${tag}${SUFFIX}/best_checkpoint.pth"
  local out="$HELDOUT_EVALUATION_ROOT/user23_test/${tag}${SUFFIX}"

  [[ -f "$ckpt" ]] || {
    echo "[ERROR] Missing FT checkpoint: $ckpt"
    exit 1
  }

  if [[ -f "$out/evaluation_summary.json" ]]; then
    echo "[SKIP Test] $tag"
    return
  fi

  mkdir -p "$out"

  echo "=== Held-out Test: $tag ==="

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
    --student PRC \
    --resume "$ckpt" \
    --distillation-type none \
    --patch_size "$PATCH_SIZE" \
    --reservoir_size "$RESERVOIR_SIZE" \
    --mixup 0 \
    --cutmix 0 \
    --no-model-ema \
    --output_dir "$out" \
    2>&1 | tee "$out/eval_stdout.log"

  [[ -f "$out/evaluation_summary.json" ]] || {
    echo "[ERROR] Missing evaluation summary: $tag"
    exit 1
  }
}

echo "========================================"
echo "PRC p32 Experiment"
echo "Mode:         $MODE"
echo "Seeds:        $SEEDS_VALUE"
echo "Methods:      CE / KD"
echo "Patch size:   $PATCH_SIZE"
echo "Reservoir:    $RESERVOIR_SIZE"
echo "========================================"

for seed in $SEEDS_VALUE; do
  [[ "$seed" =~ ^[0-9]+$ ]] || {
    echo "[ERROR] Invalid seed: $seed"
    exit 2
  }

  for method in "${METHODS[@]}"; do
    run_user1 "$seed" "$method"
    run_finetune "$seed" "$method"
    run_test "$seed" "$method"
    echo "[OK] prc_p32_${method}_seed${seed}"
  done
done

echo "========================================"
echo "[OK] PRC p32 experiments completed"
echo "========================================"
