#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

RAW_ROOT="${SHL_USER23_RAW_ROOT:-$PROJECT_ROOT/dataset/SHL_2023/validate/Hips}"
OUTPUT_ROOT="${SHL_USER23_ROOT:-$PROJECT_ROOT/dataset/2023_processed_user23/100hz_5.0s_overlap0.0s}"
PREPARER="$SCRIPT_DIR/prepare_shl2023_user23.py"

for required in "$RAW_ROOT/Acc.txt" "$RAW_ROOT/Label.txt" "$PREPARER"; do
  if [[ ! -f "$required" ]]; then
    echo "[ERROR] Missing required file: $required"
    exit 1
  fi
done

if [[ -f "$OUTPUT_ROOT/Hips_Acc.npy" && -f "$OUTPUT_ROOT/Hips_Label.npy" && "${FORCE:-0}" != "1" ]]; then
  echo "[SKIP] User2/3 validation arrays already exist:"
  echo "  $OUTPUT_ROOT"
else
  FORCE_ARGS=()
  [[ "${FORCE:-0}" == "1" ]] && FORCE_ARGS=(--force)

  python "$PREPARER" \
    --acc "$RAW_ROOT/Acc.txt" \
    --label "$RAW_ROOT/Label.txt" \
    --output-dir "$OUTPUT_ROOT" \
    --window-size 500 \
    --expected-step-ms 10 \
    "${FORCE_ARGS[@]}"
fi

python - "$OUTPUT_ROOT" <<'PY'
import sys
from pathlib import Path
import numpy as np

root = Path(sys.argv[1])
acc = np.load(root / "Hips_Acc.npy", mmap_mode="r")
label = np.load(root / "Hips_Label.npy", mmap_mode="r")
print("=== USER2/3 ARRAY CHECK ===")
print("root:", root)
print("acc:", acc.shape, acc.dtype)
print("label:", label.shape, label.dtype)
print("label_values:", np.unique(label).tolist())
print("counts_match:", len(acc) == len(label))
if acc.ndim != 3 or acc.shape[1:] != (3, 500):
    raise SystemExit(f"Unexpected Acc shape: {acc.shape}")
if label.ndim != 1 or len(acc) != len(label):
    raise SystemExit("Acc/Label mismatch")
if not np.array_equal(np.unique(label), np.arange(1, 9)):
    raise SystemExit("Expected raw labels 1..8")
PY

