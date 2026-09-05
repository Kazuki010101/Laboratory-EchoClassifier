#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

SAMPLES="${LAYER_ANALYSIS_SAMPLES:-512}"
BATCH_SIZE="${LAYER_ANALYSIS_BATCH_SIZE:-16}"
OUTPUT_DIR="$RESULTS_ROOT/layer_selection"
SPLIT_FILE="$EXPERIMENT_ROOT/data_splits/shl2023_distillation_split_indices.npz"

for FILE in \
  "$SHL_DISTILL_ACC" \
  "$SHL_DISTILL_LABEL" \
  "$TEACHER_B_CHECKPOINT" \
  "$SPLIT_FILE" \
  "$SCRIPT_DIR/analyze_senvt_layer_selection.py"
do
  [[ -f "$FILE" ]] || { echo "[ERROR] Missing: $FILE"; exit 1; }
done

if pgrep -u "$USER" -f 'python .*main.py' >/dev/null 2>&1 && \
   [[ "${ALLOW_DURING_TRAINING:-0}" != "1" ]]; then
  echo "[STOP] A main.py training process is still running."
  echo "Run this analysis after Step 6 finishes, so it does not affect training."
  echo "To inspect: ps -u \"$USER\" -o pid,etime,cmd | grep main.py"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_ROOT"

echo "=================================================="
echo "SENvT layer selection analysis for TGEC"
echo "=================================================="
echo "teacher:        $TEACHER_B_CHECKPOINT"
echo "split:          $SPLIT_FILE"
echo "samples:        $SAMPLES"
echo "batch_size:     $BATCH_SIZE"
echo "selected_layer: 1 (zero-based)"
echo "output:         $OUTPUT_DIR"
echo "=================================================="

python "$SCRIPT_DIR/analyze_senvt_layer_selection.py" \
  --acc "$SHL_DISTILL_ACC" \
  --label "$SHL_DISTILL_LABEL" \
  --checkpoint "$TEACHER_B_CHECKPOINT" \
  --split-indices "$SPLIT_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --samples "$SAMPLES" \
  --batch-size "$BATCH_SIZE" \
  --seed "$SEED" \
  --device "$DEVICE" \
  --input-size "$INPUT_SIZE" \
  --student-patch-size 16 \
  --keep-ratio 0.5 \
  --selected-layer 1 \
  2>&1 | tee "$OUTPUT_DIR/analysis_stdout.log"

echo
echo "[OK] Layer analysis completed: $OUTPUT_DIR"
echo "Main paper figure: $OUTPUT_DIR/01_layer_selection.pdf"
