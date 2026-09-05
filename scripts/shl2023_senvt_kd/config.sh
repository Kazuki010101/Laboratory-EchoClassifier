#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(
  cd "$(dirname "${BASH_SOURCE[0]}")"
  pwd
)"

PROJECT_ROOT="$(
  cd "$SCRIPT_DIR/../.."
  pwd
)"

# --------------------------------------------------
# SHL-2023 distillation data
# --------------------------------------------------

SHL_DISTILL_ROOT="$PROJECT_ROOT/dataset/2023_processed/100hz_5.0s_overlap0.0s"

SHL_DISTILL_ACC="$SHL_DISTILL_ROOT/Hips_Acc.npy"
SHL_DISTILL_LABEL="$SHL_DISTILL_ROOT/Hips_Label.npy"

# --------------------------------------------------
# SENvT-B
# --------------------------------------------------

SENVT_B_PRETRAINED="$PROJECT_ROOT/dataset/SENvT-u4/1000k_task4/best.pth"

# 今回、新しく学習するSHL用教師
EXPERIMENT_ROOT="$PROJECT_ROOT/experiments/shl2023_senvt_kd"

TEACHER_ROOT="$EXPERIMENT_ROOT/teacher_finetune"
TEACHER_B_ROOT="$TEACHER_ROOT/senvtB"

TEACHER_B_CHECKPOINT="$TEACHER_B_ROOT/best_checkpoint.pth"

# --------------------------------------------------
# Common training settings
# --------------------------------------------------

SEED=0
DEVICE="cuda"

INPUT_SIZE=496
NUM_CLASSES=8

BATCH_SIZE=64
NUM_WORKERS=8

LEARNING_RATE="1e-5"
MIN_LEARNING_RATE="1e-6"
WEIGHT_DECAY="0.05"

TEACHER_EPOCHS=100
TEACHER_WARMUP_EPOCHS=5

# --------------------------------------------------
# Distillation settings
# 後のAPS/SIR蒸留スクリプトで使用
# --------------------------------------------------

DISTILLATION_ALPHA="0.7"
DISTILLATION_TAU="2.5"

STUDENT_EPOCHS=100
STUDENT_BATCH_SIZE=128
STUDENT_LEARNING_RATE="5e-4"
STUDENT_WARMUP_EPOCHS=5

# SHL-2023 official validation (User2/User3) downstream fine-tuning
SHL_USER23_ROOT="$PROJECT_ROOT/dataset/2023_processed_user23/100hz_5.0s_overlap0.0s"
USER23_SPLIT="$EXPERIMENT_ROOT/data_splits/shl2023_user23_downstream_split_indices.npz"
DOWNSTREAM_EPOCHS="${DOWNSTREAM_EPOCHS:-100}"
DOWNSTREAM_BATCH_SIZE="${DOWNSTREAM_BATCH_SIZE:-64}"
DOWNSTREAM_LEARNING_RATE="${DOWNSTREAM_LEARNING_RATE:-5e-4}"
DOWNSTREAM_WARMUP_EPOCHS="${DOWNSTREAM_WARMUP_EPOCHS:-5}"

# --------------------------------------------------
# Output directories
# --------------------------------------------------

STUDENT_BASELINE_ROOT="$EXPERIMENT_ROOT/student_baseline"
STUDENT_DISTILLATION_ROOT="$EXPERIMENT_ROOT/student_distillation"
STUDENT_FINETUNE_ROOT="$EXPERIMENT_ROOT/student_finetune"
HELDOUT_EVALUATION_ROOT="$EXPERIMENT_ROOT/heldout_evaluation"
RESULTS_ROOT="$EXPERIMENT_ROOT/results"

mkdir -p "$TEACHER_ROOT"
mkdir -p "$STUDENT_BASELINE_ROOT"
mkdir -p "$STUDENT_DISTILLATION_ROOT"
mkdir -p "$STUDENT_FINETUNE_ROOT"
mkdir -p "$HELDOUT_EVALUATION_ROOT"
mkdir -p "$RESULTS_ROOT"

# --------------------------------------------------
# GPU energy measurement
# --------------------------------------------------

GPU_INDEX="${GPU_INDEX:-0}"
ENERGY_INTERVAL_MS="${ENERGY_INTERVAL_MS:-500}"

ENERGY_RUNNER="$SCRIPT_DIR/run_with_gpu_energy.sh"
ENERGY_SUMMARIZER="$SCRIPT_DIR/summarize_gpu_energy.py"

export GPU_INDEX
export ENERGY_INTERVAL_MS
export ENERGY_RUNNER
export ENERGY_SUMMARIZER
