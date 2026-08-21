#!/usr/bin/env bash
set -euo pipefail

# このshは transfer_scripts/ 配下に置く想定。
# どこから実行しても EchoAttnNet のプロジェクトルートへ移動してから実行する。
#
# 実行例:
#   bash transfer_scripts/run_pamap2_footprint_memory.sh
#
# GPUで測る場合:
#   DEVICE=cuda bash transfer_scripts/run_pamap2_footprint_memory.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

EXP_ROOT="experiments/SHL2023_senvt_echo_w496"
OUT_DIR="${EXP_ROOT}/pamap2_compare_all_models/footprint"
CSV_PATH="${OUT_DIR}/model_static_metrics_with_memory.csv"

DATASET="PAMAP2"
INPUT_SIZE=496
CHANNELS=3
NUM_CLASSES=8
BATCH_SIZE=1

# 論文・概要に載せるなら、まずCPU推論で統一するのが安全。
DEVICE="${DEVICE:-cpu}"

WARMUP=30
REPEAT=200
MEMORY_WARMUP=10
MEMORY_REPEAT=200

mkdir -p "${OUT_DIR}"

if [ -f "${CSV_PATH}" ]; then
  BACKUP_PATH="${CSV_PATH}.bak.$(date +%Y%m%d_%H%M%S)"
  echo "[info] Existing CSV found. Backup to ${BACKUP_PATH}"
  mv "${CSV_PATH}" "${BACKUP_PATH}"
fi

run_metric () {
  local RUN_NAME="$1"
  local MODEL="$2"
  shift 2

  echo "============================================================"
  echo "[run] ${RUN_NAME}"
  echo "============================================================"

  python model4edgetest.py \
    --dataset "${DATASET}" \
    --model "${MODEL}" \
    --run-name "${RUN_NAME}" \
    --input-size "${INPUT_SIZE}" \
    --channels "${CHANNELS}" \
    --num-classes "${NUM_CLASSES}" \
    --batch-size "${BATCH_SIZE}" \
    --device "${DEVICE}" \
    --warmup "${WARMUP}" \
    --repeat "${REPEAT}" \
    --memory-warmup "${MEMORY_WARMUP}" \
    --memory-repeat "${MEMORY_REPEAT}" \
    --csv-path "${CSV_PATH}" \
    "$@"
}

echo "============================================================"
echo "[info] Measuring all PAMAP2 comparison models"
echo "[info] device: ${DEVICE}"
echo "[info] csv: ${CSV_PATH}"
echo "============================================================"

# =========================
# Large / neural baselines
# =========================

run_metric "MLPMixer" "MLPMixer"

run_metric "Resnet_S" "Resnet_S"
run_metric "Resnet_M" "Resnet_M"
run_metric "Resnet_L" "Resnet_L"

run_metric "DeepConvLSTM25" "DeepConvLSTM25"
run_metric "DeepConvLSTM50" "DeepConvLSTM50"
run_metric "DeepConvLSTM100" "DeepConvLSTM100"

# =========================
# PRC
# =========================

run_metric "PRC_p32_r1000" "PRC" \
  --patch-size 32 \
  --reservoir-size 1000

run_metric "PRC_p64_r1000" "PRC" \
  --patch-size 64 \
  --reservoir-size 1000

run_metric "PRC_p128_r1000" "PRC" \
  --patch-size 128 \
  --reservoir-size 1000

run_metric "PRC_p128_r4000" "PRC" \
  --patch-size 128 \
  --reservoir-size 4000

# =========================
# PESAC
# =========================

run_metric "PESAC_p32_r1000" "PESAC" \
  --patch-size 32 \
  --reservoir-size 1000

run_metric "PESAC_p64_r1000" "PESAC" \
  --patch-size 64 \
  --reservoir-size 1000

run_metric "PESAC_p128_r1000" "PESAC" \
  --patch-size 128 \
  --reservoir-size 1000

run_metric "PESAC_p128_r4000" "PESAC" \
  --patch-size 128 \
  --reservoir-size 4000

# =========================
# LRGR
# =========================

run_metric "LRGR_p32_r1000_rank32" "PRC_LRGR" \
  --patch-size 32 \
  --reservoir-size 1000 \
  --reservoir-rank 32

run_metric "LRGR_p64_r1000_rank32" "PRC_LRGR" \
  --patch-size 64 \
  --reservoir-size 1000 \
  --reservoir-rank 32

run_metric "LRGR_p128_r1000_rank32" "PRC_LRGR" \
  --patch-size 128 \
  --reservoir-size 1000 \
  --reservoir-rank 32

run_metric "LRGR_p128_r4000_rank64" "PRC_LRGR" \
  --patch-size 128 \
  --reservoir-size 4000 \
  --reservoir-rank 64

# =========================
# APS-LRGR
# =========================

run_metric "APS_LRGR_p32_r1000_rank32_keep050" "APS_LRGR" \
  --patch-size 32 \
  --reservoir-size 1000 \
  --reservoir-rank 32 \
  --patch-keep-ratio 0.50

run_metric "APS_LRGR_p64_r1000_rank32_keep050" "APS_LRGR" \
  --patch-size 64 \
  --reservoir-size 1000 \
  --reservoir-rank 32 \
  --patch-keep-ratio 0.50

run_metric "APS_LRGR_p128_r1000_rank32_keep067" "APS_LRGR" \
  --patch-size 128 \
  --reservoir-size 1000 \
  --reservoir-rank 32 \
  --patch-keep-ratio 0.67

run_metric "APS_LRGR_p128_r4000_rank64_keep067" "APS_LRGR" \
  --patch-size 128 \
  --reservoir-size 4000 \
  --reservoir-rank 64 \
  --patch-keep-ratio 0.67


# =========================
# IFSA-LRGR
# =========================

run_metric "IFSA_LRGR_p64_r1000_rank32_irank16" "IFSA_LRGR" \
  --patch-size 64 \
  --reservoir-size 1000 \
  --reservoir-rank 32 \
  --input-rank 16

run_metric "IFSA_LRGR_p128_r1000_rank32_irank16" "IFSA_LRGR" \
  --patch-size 128 \
  --reservoir-size 1000 \
  --reservoir-rank 32 \
  --input-rank 16

run_metric "IFSA_LRGR_p128_r1000_rank32_irank32" "IFSA_LRGR" \
  --patch-size 128 \
  --reservoir-size 1000 \
  --reservoir-rank 32 \
  --input-rank 32

run_metric "IFSA_LRGR_p128_r4000_rank64_irank32" "IFSA_LRGR" \
  --patch-size 128 \
  --reservoir-size 4000 \
  --reservoir-rank 64 \
  --input-rank 32

# =========================
# SIR-LRGR
# =========================

run_metric "SIR_LRGR_p32_r1000_rank32_target050" "SIR_LRGR" \
  --patch-size 32 \
  --reservoir-size 1000 \
  --reservoir-rank 32 \
  --innovation-target-ratio 0.50 \
  --innovation-threshold 0.50 \
  --innovation-hidden-dim 64 \
  --innovation-min-keep 1

run_metric "SIR_LRGR_p64_r1000_rank32_target050" "SIR_LRGR" \
  --patch-size 64 \
  --reservoir-size 1000 \
  --reservoir-rank 32 \
  --innovation-target-ratio 0.50 \
  --innovation-threshold 0.50 \
  --innovation-hidden-dim 64 \
  --innovation-min-keep 1

run_metric "SIR_LRGR_p128_r1000_rank32_target067" "SIR_LRGR" \
  --patch-size 128 \
  --reservoir-size 1000 \
  --reservoir-rank 32 \
  --innovation-target-ratio 0.67 \
  --innovation-threshold 0.50 \
  --innovation-hidden-dim 64 \
  --innovation-min-keep 1

echo "============================================================"
echo "[done] CSV saved to:"
echo "  ${CSV_PATH}"
echo "============================================================"

echo ""
echo "[preview]"
if command -v column >/dev/null 2>&1; then
  column -s, -t "${CSV_PATH}" | head -n 25
else
  head -n 25 "${CSV_PATH}"
fi