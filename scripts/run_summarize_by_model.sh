#!/bin/bash
set -eo pipefail

source scripts/config_senvt_echo.sh

python summarize_by_model.py \
  --input-csv "$EXP_ROOT/final_summary.csv" \
  --output-csv "$EXP_ROOT/final_summary_by_model.csv"