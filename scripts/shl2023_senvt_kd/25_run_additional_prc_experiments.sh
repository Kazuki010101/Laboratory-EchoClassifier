#!/usr/bin/env bash
set -euo pipefail

# Additional end-to-end experiments without overwriting the existing seed0/k16 runs.
#
# Usage:
#   bash 25_run_additional_prc_experiments.sh multiseed full core
#   bash 25_run_additional_prc_experiments.sh keep24    full core
#
# Optional:
#   SEEDS="1 2" RESUME_PARTIAL=1 bash ... multiseed full core
#   SEEDS="0 1 2" bash ... keep24 full core
#
# scope=core:
#   multiseed -> PRC-KD, APS-KD, TG-Skip, TGEC
#   keep24    -> APS-KD, TG-Skip, TGEC
# scope=all additionally includes PRC-CE/APS-CE where applicable.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

ACTION="${1:-multiseed}"
MODE="${2:-full}"
SCOPE="${3:-core}"

case "$MODE" in
  smoke)
    USER1_EPOCHS=1
    USER1_WARMUP=0
    FT_EPOCHS=1
    FT_WARMUP=0
    RUN_SUFFIX="_smoke"
    ;;
  full)
    USER1_EPOCHS="$STUDENT_EPOCHS"
    USER1_WARMUP="$STUDENT_WARMUP_EPOCHS"
    FT_EPOCHS="$DOWNSTREAM_EPOCHS"
    FT_WARMUP="$DOWNSTREAM_WARMUP_EPOCHS"
    RUN_SUFFIX=""
    ;;
  *)
    echo "[ERROR] MODE must be smoke or full: $MODE"
    exit 2
    ;;
esac

case "$SCOPE" in
  core|all) ;;
  *)
    echo "[ERROR] SCOPE must be core or all: $SCOPE"
    exit 2
    ;;
esac

case "$ACTION" in
  multiseed)
    PATCH_SIZE_VALUE=16
    KEEP_RATIO_VALUE=0.5
    VARIANT=""
    SEEDS_VALUE="${SEEDS:-1 2}"
    if [[ "$SCOPE" == "all" ]]; then
      METHODS=(prc_ce prc_kd aps_ce aps_kd tg_skip tgec)
    else
      METHODS=(prc_kd aps_kd tg_skip tgec)
    fi
    ;;
  keep24)
    PATCH_SIZE_VALUE=16
    KEEP_RATIO_VALUE=0.75
    VARIANT="ps16_k24"
    SEEDS_VALUE="${SEEDS:-0}"
    if [[ "$SCOPE" == "all" ]]; then
      METHODS=(aps_ce aps_kd tg_skip tgec)
    else
      METHODS=(aps_kd tg_skip tgec)
    fi
    ;;
  *)
    echo "[ERROR] ACTION must be multiseed or keep24: $ACTION"
    exit 2
    ;;
esac

require_file() {
  [[ -f "$1" ]] || { echo "[ERROR] Missing: $1"; exit 1; }
}

require_file "$SHL_DISTILL_ACC"
require_file "$SHL_DISTILL_LABEL"
require_file "$SHL_USER23_ROOT/Hips_Acc.npy"
require_file "$SHL_USER23_ROOT/Hips_Label.npy"
require_file "$USER23_SPLIT"
require_file "$TEACHER_B_CHECKPOINT"

cd "$PROJECT_ROOT"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

tag_for() {
  local method="$1"
  local seed="$2"
  if [[ -n "$VARIANT" ]]; then
    printf '%s_%s_seed%s' "$method" "$VARIANT" "$seed"
  else
    printf '%s_seed%s' "$method" "$seed"
  fi
}

student_for() {
  case "$1" in
    prc_ce|prc_kd) echo "PRC" ;;
    aps_ce|aps_kd) echo "APS_PRC" ;;
    tg_skip) echo "TG_SKIP_PRC" ;;
    tgec) echo "TGEC_PRC" ;;
    *) echo "[ERROR] Unknown method: $1" >&2; return 2 ;;
  esac
}

user1_output_for() {
  local method="$1"
  local seed="$2"
  local tag
  tag="$(tag_for "$method" "$seed")"
  case "$method" in
    prc_ce) echo "$STUDENT_BASELINE_ROOT/PRC_ce_seed${seed}${RUN_SUFFIX}" ;;
    prc_kd) echo "$STUDENT_DISTILLATION_ROOT/standard_kd/PRC_seed${seed}${RUN_SUFFIX}" ;;
    *) echo "$STUDENT_DISTILLATION_ROOT/prc_condensation/${tag}${RUN_SUFFIX}" ;;
  esac
}

resume_args_for() {
  local out="$1"
  RESUME_ARGS=()
  if [[ -f "$out/checkpoint.pth" && "${FORCE:-0}" != "1" ]]; then
    if [[ "${RESUME_PARTIAL:-0}" == "1" ]]; then
      RESUME_ARGS=(--resume "$out/checkpoint.pth")
      echo "[RESUME] $out"
    else
      echo "[STOP] Partial run found: $out"
      echo "       Resume with RESUME_PARTIAL=1"
      exit 1
    fi
  fi
}

run_user1() {
  local method="$1"
  local seed="$2"
  local student distill route_weight content_weight out
  student="$(student_for "$method")"
  distill="none"
  route_weight=0
  content_weight=0

  case "$method" in
    prc_kd|aps_kd|tg_skip|tgec) distill="soft" ;;
  esac
  [[ "$method" == "tg_skip" || "$method" == "tgec" ]] && route_weight=1
  [[ "$method" == "tgec" ]] && content_weight=1

  out="$(user1_output_for "$method" "$seed")"
  if [[ -f "$out/summary.json" && "${FORCE:-0}" != "1" ]]; then
    echo "[SKIP User1] $out"
    return
  fi

  resume_args_for "$out"
  mkdir -p "$out"
  echo "=== User1: method=$method seed=$seed keep_ratio=$KEEP_RATIO_VALUE output=$out ==="

  cmd=(python main.py
    --data SHL2023
    --input-size "$INPUT_SIZE"
    --epochs "$USER1_EPOCHS"
    --batch-size "$STUDENT_BATCH_SIZE"
    --num_workers "$NUM_WORKERS"
    --device "$DEVICE"
    --seed "$seed"
    --student "$student"
    --distillation-type "$distill"
    --distillation-alpha "$DISTILLATION_ALPHA"
    --distillation-tau "$DISTILLATION_TAU"
    --patch_size "$PATCH_SIZE_VALUE"
    --reservoir_size 1000
    --reservoir_rank 64
    --patch_keep_ratio "$KEEP_RATIO_VALUE"
    --input_rank 16
    --tgec-teacher-layer 1
    --tgec-route-weight "$route_weight"
    --tgec-content-weight "$content_weight"
    --mixup 0 --cutmix 0 --smoothing 0.1
    --clip-grad 1 --opt adamw --weight-decay "$WEIGHT_DECAY"
    --lr "$STUDENT_LEARNING_RATE" --min-lr "$MIN_LEARNING_RATE"
    --warmup-epochs "$USER1_WARMUP"
    --no-model-ema --unscale-lr
    --output_dir "$out")

  if [[ "$distill" != "none" ]]; then
    cmd+=(--model senvt-B --teacher-path "$TEACHER_B_CHECKPOINT")
  fi
  cmd+=("${RESUME_ARGS[@]}")
  "${cmd[@]}" 2>&1 | tee -a "$out/train_stdout.log"
}

run_finetune() {
  local method="$1"
  local seed="$2"
  local student tag source out
  student="$(student_for "$method")"
  tag="$(tag_for "$method" "$seed")"
  source="$(user1_output_for "$method" "$seed")/best_checkpoint.pth"
  out="$STUDENT_FINETUNE_ROOT/user23/${tag}${RUN_SUFFIX}"
  require_file "$source"

  if [[ -f "$out/summary.json" && "${FORCE:-0}" != "1" ]]; then
    echo "[SKIP fine-tuning] $out"
    return
  fi

  resume_args_for "$out"
  mkdir -p "$out"
  echo "=== User2/3 FT: method=$method seed=$seed keep_ratio=$KEEP_RATIO_VALUE output=$out ==="

  python main.py \
    --data SHL2023_user23_finetune --data-path "$SHL_USER23_ROOT" \
    --split-indices "$USER23_SPLIT" --finetune "$source" \
    --input-size "$INPUT_SIZE" --epochs "$FT_EPOCHS" \
    --batch-size "$DOWNSTREAM_BATCH_SIZE" --num_workers "$NUM_WORKERS" \
    --device "$DEVICE" --seed "$seed" --student "$student" \
    --distillation-type none --patch_size "$PATCH_SIZE_VALUE" \
    --reservoir_size 1000 --reservoir_rank 64 \
    --patch_keep_ratio "$KEEP_RATIO_VALUE" --input_rank 16 \
    --mixup 0 --cutmix 0 --smoothing 0.1 --clip-grad 1 \
    --opt adamw --weight-decay "$WEIGHT_DECAY" \
    --lr "$DOWNSTREAM_LEARNING_RATE" --min-lr "$MIN_LEARNING_RATE" \
    --warmup-epochs "$FT_WARMUP" --no-model-ema --unscale-lr \
    --output_dir "$out" "${RESUME_ARGS[@]}" \
    2>&1 | tee -a "$out/train_stdout.log"
}

run_test() {
  local method="$1"
  local seed="$2"
  local student tag ckpt out
  student="$(student_for "$method")"
  tag="$(tag_for "$method" "$seed")"
  ckpt="$STUDENT_FINETUNE_ROOT/user23/${tag}${RUN_SUFFIX}/best_checkpoint.pth"
  out="$HELDOUT_EVALUATION_ROOT/user23_test/${tag}${RUN_SUFFIX}"
  require_file "$ckpt"

  if [[ -f "$out/evaluation_summary.json" && "${FORCE:-0}" != "1" ]]; then
    echo "[SKIP test] $out"
    return
  fi

  mkdir -p "$out"
  echo "=== Held-out test: method=$method seed=$seed output=$out ==="
  python main.py --eval \
    --data SHL2023_user23_test --data-path "$SHL_USER23_ROOT" \
    --split-indices "$USER23_SPLIT" --input-size "$INPUT_SIZE" \
    --batch-size "$DOWNSTREAM_BATCH_SIZE" --num_workers "$NUM_WORKERS" \
    --device "$DEVICE" --seed "$seed" --student "$student" \
    --resume "$ckpt" --distillation-type none \
    --patch_size "$PATCH_SIZE_VALUE" --reservoir_size 1000 --reservoir_rank 64 \
    --patch_keep_ratio "$KEEP_RATIO_VALUE" --input_rank 16 \
    --mixup 0 --cutmix 0 --no-model-ema --output_dir "$out" \
    2>&1 | tee "$out/eval_stdout.log"
}

echo "=================================================="
echo "Additional PRC experiments"
echo "action:      $ACTION"
echo "mode:        $MODE"
echo "scope:       $SCOPE"
echo "seeds:       $SEEDS_VALUE"
echo "patch size:  $PATCH_SIZE_VALUE"
echo "keep ratio:  $KEEP_RATIO_VALUE"
echo "variant:     ${VARIANT:-standard}"
echo "methods:     ${METHODS[*]}"
echo "split:       $USER23_SPLIT (reused; not regenerated)"
echo "=================================================="

for seed in $SEEDS_VALUE; do
  [[ "$seed" =~ ^[0-9]+$ ]] || { echo "[ERROR] Invalid seed: $seed"; exit 2; }
  for method in "${METHODS[@]}"; do
    run_user1 "$method" "$seed"
    run_finetune "$method" "$seed"
    run_test "$method" "$seed"
  done
done

echo "[OK] All requested additional experiments completed."

