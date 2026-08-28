#!/bin/bash
set -euo pipefail

source transfer_scripts/config_pamap2_all_models.sh

MODEL_FILTER="PRC_p32_r1000,LRGR_p32_r1000_rank32,APS_LRGR_p32_r1000_rank32_keep050,SIR_LRGR_p32_r1000_rank32_target050,PRC_p64_r1000,LRGR_p64_r1000_rank32,APS_LRGR_p64_r1000_rank32_keep050,SIR_LRGR_p64_r1000_rank32_target050"

OUT_DIR="$EXP_ROOT/pamap2_efficiency"
BASE_SUMMARY="${BASE_SUMMARY:-$OUT_DIR/word_prc_lrgr_aps_sir_p32_p64_efficiency_cpu_summary.csv}"
ENERGY_CSV="$OUT_DIR/word_prc_lrgr_aps_sir_p32_p64_cpu_energy_only.csv"
ENERGY_SUMMARY="$OUT_DIR/word_prc_lrgr_aps_sir_p32_p64_cpu_energy_only_summary.csv"
MERGED_SUMMARY="$OUT_DIR/word_prc_lrgr_aps_sir_p32_p64_efficiency_cpu_summary_with_energy.csv"
RAPL_USE_SUDO="${RAPL_USE_SUDO:-0}"

if [[ ! -f "$BASE_SUMMARY" ]]; then
  echo "[error] base summary not found: $BASE_SUMMARY" >&2
  exit 1
fi

if ! compgen -G '/sys/class/powercap/intel-rapl:*/energy_uj' > /dev/null; then
  echo "[error] package-level RAPL counters are not visible." >&2
  echo "Run on the host or expose /sys/class/powercap to the container." >&2
  exit 1
fi

RAPL_PROBE="$(compgen -G '/sys/class/powercap/intel-rapl:*/energy_uj' | head -n 1)"
if [[ "$RAPL_USE_SUDO" == "1" ]]; then
  if ! sudo -n cat "$RAPL_PROBE" > /dev/null; then
    echo "[error] RAPL cannot be read with passwordless sudo: $RAPL_PROBE" >&2
    exit 1
  fi
elif [[ ! -r "$RAPL_PROBE" ]]; then
  echo "[error] RAPL is visible but is not readable: $RAPL_PROBE" >&2
  echo "If 'sudo -n cat $RAPL_PROBE' prints a number, rerun with RAPL_USE_SUDO=1." >&2
  exit 1
fi

export RAPL_USE_SUDO

mkdir -p "$OUT_DIR"

python evaluate_efficiency.py \
  --experiments-root "$PAMAP2_BASELINE_EXP_ROOT" \
  --experiments-root "$EXP_ROOT/pamap2_transfer" \
  --experiments-root "$PAMAP2_TRANSFER_EXP_ROOT" \
  --pamap2-root "$PAMAP2_ROOT" \
  --device cpu \
  --model-filter "$MODEL_FILTER" \
  --latency-samples 500 \
  --skip-latency \
  --skip-memory \
  --skip-thop \
  --measure-energy \
  --energy-seconds 30 \
  --idle-seconds 10 \
  --output-csv "$ENERGY_CSV" \
  --summary-csv "$ENERGY_SUMMARY" \
  --overwrite

python transfer_scripts/merge_efficiency_energy_summary.py \
  --base "$BASE_SUMMARY" \
  --energy "$ENERGY_SUMMARY" \
  --output "$MERGED_SUMMARY"

echo "===== ENERGY SUBJECT ROWS: $ENERGY_CSV ====="
echo "===== ENERGY SUMMARY: $ENERGY_SUMMARY ====="
echo "===== MERGED SUMMARY: $MERGED_SUMMARY ====="
