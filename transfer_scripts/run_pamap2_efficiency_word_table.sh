#!/bin/bash
set -euo pipefail

source transfer_scripts/config_pamap2_all_models.sh

DEVICE="${DEVICE:-cpu}"
MEASURE_ENERGY="${MEASURE_ENERGY:-1}"
RESUME_OUTPUT="${RESUME_OUTPUT:-0}"
MODEL_FILTER="PRC_p32_r1000,LRGR_p32_r1000_rank32,APS_LRGR_p32_r1000_rank32_keep050,SIR_LRGR_p32_r1000_rank32_target050,PRC_p64_r1000,LRGR_p64_r1000_rank32,APS_LRGR_p64_r1000_rank32_keep050,SIR_LRGR_p64_r1000_rank32_target050"

MODE="efficiency"
ENERGY_ARGS=()
if [[ "$MEASURE_ENERGY" == "1" ]]; then
  MODE="energy"
  ENERGY_ARGS=(
    --measure-energy
    --energy-seconds 30
    --idle-seconds 10
  )
fi

OUTPUT_ARGS=(--overwrite)
if [[ "$RESUME_OUTPUT" == "1" ]]; then
  OUTPUT_ARGS=(--resume-output)
fi

OUT_DIR="$EXP_ROOT/pamap2_efficiency"
RUN_NAME="word_prc_lrgr_aps_sir_p32_p64_${MODE}_${DEVICE}"
OUT_CSV="$OUT_DIR/${RUN_NAME}.csv"
SUMMARY_CSV="$OUT_DIR/${RUN_NAME}_summary.csv"
WORD_CSV="$OUT_DIR/${RUN_NAME}_word_table.csv"

mkdir -p "$OUT_DIR"

python evaluate_efficiency.py \
  --experiments-root "$PAMAP2_BASELINE_EXP_ROOT" \
  --experiments-root "$EXP_ROOT/pamap2_transfer" \
  --experiments-root "$PAMAP2_TRANSFER_EXP_ROOT" \
  --pamap2-root "$PAMAP2_ROOT" \
  --device "$DEVICE" \
  --model-filter "$MODEL_FILTER" \
  --latency-samples 500 \
  --latency-repeats 3 \
  --warmup 50 \
  "${ENERGY_ARGS[@]}" \
  --output-csv "$OUT_CSV" \
  --summary-csv "$SUMMARY_CSV" \
  --word-table-csv "$WORD_CSV" \
  "${OUTPUT_ARGS[@]}"

echo "===== SUBJECT ROWS: $OUT_CSV ====="
echo "===== SUBJECT SUMMARY: $SUMMARY_CSV ====="
echo "===== WORD TABLE: $WORD_CSV ====="
