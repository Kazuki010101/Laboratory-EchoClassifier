#!/bin/bash
set -eo pipefail

source scripts/config_senvt_echo.sh

python summarize_echo_results.py \
  --exp-root "$EXP_ROOT" \
  --footprint-csv "$EXP_ROOT/footprint/model_static_metrics.csv" \
  --output-csv "$EXP_ROOT/final_summary.csv"