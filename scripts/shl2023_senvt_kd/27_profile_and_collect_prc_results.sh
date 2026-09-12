#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PROFILE_REPEATS_VALUE="${PROFILE_REPEATS:-200}"
PROFILE_WARMUP_VALUE="${PROFILE_WARMUP:-20}"
PROFILE_CPU_THREADS_VALUE="${PROFILE_CPU_THREADS:-1}"
PROFILE_RUN_GPU_VALUE="${PROFILE_RUN_GPU:-1}"
PROFILE_RUN_CPU_VALUE="${PROFILE_RUN_CPU:-1}"

COMMON_PROFILE_ARGS=(
  --input-size "$INPUT_SIZE"
  --reservoir-size 1000
  --patch-size 16
  --keep-ratio 0.5
  --warmup "$PROFILE_WARMUP_VALUE"
  --repeats "$PROFILE_REPEATS_VALUE"
)

if [[ "$PROFILE_RUN_GPU_VALUE" == "1" ]]; then
  if python -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
    echo "=== PRC efficiency profiling: GPU, patch size 16, keep 16/31 ==="
    python "$SCRIPT_DIR/profile_prc_models_v2.py" \
      --output "$RESULTS_ROOT/prc_efficiency_ps16_k16_gpu.json" \
      --device cuda \
      "${COMMON_PROFILE_ARGS[@]}"
  else
    echo "[WARN] CUDA is unavailable; GPU profiling was skipped." >&2
  fi
fi

if [[ "$PROFILE_RUN_CPU_VALUE" == "1" ]]; then
  echo "=== PRC efficiency profiling: CPU, patch size 16, keep 16/31 ==="
  OMP_NUM_THREADS="$PROFILE_CPU_THREADS_VALUE" \
  MKL_NUM_THREADS="$PROFILE_CPU_THREADS_VALUE" \
  OPENBLAS_NUM_THREADS="$PROFILE_CPU_THREADS_VALUE" \
  python "$SCRIPT_DIR/profile_prc_models_v2.py" \
    --output "$RESULTS_ROOT/prc_efficiency_ps16_k16_cpu.json" \
    --device cpu \
    --cpu-threads "$PROFILE_CPU_THREADS_VALUE" \
    "${COMMON_PROFILE_ARGS[@]}"
fi

echo "=== Collecting classification results ==="
python "$SCRIPT_DIR/collect_prc_results.py" \
  --experiment-root "$EXPERIMENT_ROOT" \
  --output-dir "$RESULTS_ROOT"

echo "[OK] GPU profile: $RESULTS_ROOT/prc_efficiency_ps16_k16_gpu.json"
echo "[OK] CPU profile: $RESULTS_ROOT/prc_efficiency_ps16_k16_cpu.json"

