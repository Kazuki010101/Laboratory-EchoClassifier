
#!/usr/bin/env bash
set -euo pipefail

# ==========================================
# PRC-p32 Computational Efficiency Profiling
# ==========================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

PROFILE_SCRIPT="$SCRIPT_DIR/profile_prc_models_v2.py"
OUTPUT_DIR="$PROJECT_ROOT/experiments/shl2023_senvt_kd/efficiency_prc_p32"

mkdir -p "$OUTPUT_DIR"

# Settings: Same as the paper
INPUT_SIZE=496
RESERVOIR_SIZE=1000
NUM_CLASSES=8
PATCH_SIZE=32
KEEP_RATIO=0.5

BATCH_SIZE=1
WARMUP=20
REPEATS=200
CPU_THREADS=1

if [[ ! -f "$PROFILE_SCRIPT" ]]; then
    echo "[ERROR] Profiling script not found: $PROFILE_SCRIPT"
    exit 1
fi

# Activate conda environment
if [[ "${CONDA_DEFAULT_ENV:-}" != "mfm311" ]]; then
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        conda activate mfm311
    else
        echo "[ERROR] Conda not found."
        exit 1
    fi
fi

echo "=========================================="
echo "PRC-p32 Computational Efficiency Profiling"
echo "Python: $(which python)"
echo "Environment: ${CONDA_DEFAULT_ENV:-unknown}"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# ==========================================
# 1. CPU profiling
# ==========================================

echo ""
echo "[1/3] CPU profiling..."

python "$PROFILE_SCRIPT" \
    --output "$OUTPUT_DIR/prc_p32_cpu.json" \
    --device cpu \
    --input-size "$INPUT_SIZE" \
    --reservoir-size "$RESERVOIR_SIZE" \
    --num-classes "$NUM_CLASSES" \
    --patch-size "$PATCH_SIZE" \
    --keep-ratio "$KEEP_RATIO" \
    --batch-size "$BATCH_SIZE" \
    --warmup "$WARMUP" \
    --repeats "$REPEATS" \
    --cpu-threads "$CPU_THREADS" \
    > "$OUTPUT_DIR/cpu.log" 2>&1

echo "[OK] CPU profiling completed."

# ==========================================
# 2. GPU profiling
# ==========================================

echo ""
echo "[2/3] Checking CUDA availability..."

if python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable"
x = torch.randn(32, 32, device="cuda")
y = x @ x
torch.cuda.synchronize()
print("GPU:", torch.cuda.get_device_name(0))
PY
then
    echo "[2/3] GPU profiling..."

    if python "$PROFILE_SCRIPT" \
        --output "$OUTPUT_DIR/prc_p32_gpu.json" \
        --device cuda \
        --input-size "$INPUT_SIZE" \
        --reservoir-size "$RESERVOIR_SIZE" \
        --num-classes "$NUM_CLASSES" \
        --patch-size "$PATCH_SIZE" \
        --keep-ratio "$KEEP_RATIO" \
        --batch-size "$BATCH_SIZE" \
        --warmup "$WARMUP" \
        --repeats "$REPEATS" \
        > "$OUTPUT_DIR/gpu.log" 2>&1
    then
        echo "[OK] GPU profiling completed."
    else
        echo "[WARN] GPU profiling failed. Check gpu.log."
    fi
else
    echo "[WARN] CUDA unavailable. Skipping GPU profiling."
fi

# ==========================================
# 3. Summarize results
# ==========================================

echo ""
echo "[3/3] Summary"
echo "=========================================="

python - "$OUTPUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])

keys = [
    ("recurrent_tokens_including_cls_dist", "Reservoir updates"),
    ("estimated_dominant_mmacs_per_sample", "Estimated MACs (M)"),
    ("profiled_mflops_per_sample", "Profiled FLOPs (M)"),
    ("trainable_parameters", "Trainable parameters"),
    ("model_state_size_mb", "Footprint (MiB)"),
    ("latency_ms_median", "Median latency (ms)"),
]

for device in ("cpu", "gpu"):
    path = root / f"prc_p32_{device}.json"

    if not path.exists():
        print(f"\n[{device.upper()}] No results")
        continue

    data = json.loads(path.read_text())
    result = data["prc_full"]

    print(f"\n[{device.upper()}] PRC-p32")
    print("-" * 40)

    for key, label in keys:
        value = result.get(key, "N/A")
        if isinstance(value, float):
            print(f"{label:<26}: {value:.4f}")
        else:
            print(f"{label:<26}: {value}")

print("\nSaved in:", root)
PY

echo ""
echo "[DONE] PRC-p32 profiling finished."
