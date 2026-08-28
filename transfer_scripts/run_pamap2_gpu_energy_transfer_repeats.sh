#!/bin/bash
set -euo pipefail

source transfer_scripts/config_pamap2_all_models.sh

REPEATS="${REPEATS:-5}"
OUT_DIR="$EXP_ROOT/pamap2_efficiency/gpu_energy_transfer_repeats"
MODEL_FILTER="PRC_p32_r1000,LRGR_p32_r1000_rank32,APS_LRGR_p32_r1000_rank32_keep050,SIR_LRGR_p32_r1000_rank32_target050,PRC_p64_r1000,LRGR_p64_r1000_rank32,APS_LRGR_p64_r1000_rank32_keep050,SIR_LRGR_p64_r1000_rank32_target050"

mkdir -p "$OUT_DIR"

{
  echo "started_at=$(date --iso-8601=seconds)"
  echo "hostname=$(hostname)"
  echo "repeats=$REPEATS"
  echo "model_filter=$MODEL_FILTER"
  python --version
  nvidia-smi --query-gpu=index,name,uuid,driver_version,pstate,temperature.gpu,power.draw,power.limit,memory.used --format=csv
} > "$OUT_DIR/environment_before.txt"

for repeat_index in $(seq 1 "$REPEATS"); do
  repeat_tag=$(printf "%02d" "$repeat_index")
  output_csv="$OUT_DIR/transfer_gpu_energy_repeat${repeat_tag}.csv"
  summary_csv="$OUT_DIR/transfer_gpu_energy_repeat${repeat_tag}_summary.csv"

  if [[ -s "$summary_csv" ]]; then
    echo "[skip] repeat $repeat_tag already completed: $summary_csv"
    continue
  fi

  echo "===== repeat $repeat_tag / $REPEATS ====="
  date --iso-8601=seconds

  python evaluate_efficiency.py \
    --experiments-root "$EXP_ROOT/pamap2_transfer" \
    --experiments-root "$PAMAP2_TRANSFER_EXP_ROOT" \
    --pamap2-root "$PAMAP2_ROOT" \
    --device cuda \
    --model-filter "$MODEL_FILTER" \
    --latency-samples 500 \
    --latency-repeats 3 \
    --warmup 50 \
    --measure-energy \
    --energy-seconds 30 \
    --idle-seconds 10 \
    --output-csv "$output_csv" \
    --summary-csv "$summary_csv" \
    --overwrite
done

{
  echo "finished_at=$(date --iso-8601=seconds)"
  nvidia-smi --query-gpu=index,name,uuid,driver_version,pstate,temperature.gpu,power.draw,power.limit,memory.used --format=csv
} > "$OUT_DIR/environment_after.txt"

echo "===== ALL REPEATS COMPLETED ====="
echo "outputs: $OUT_DIR"
