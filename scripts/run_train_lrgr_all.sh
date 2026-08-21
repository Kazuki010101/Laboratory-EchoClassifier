#!/bin/bash
set -eo pipefail

echo "===== Train LRGR Baseline ====="
bash scripts/run_baseline_lrgr.sh

echo "===== Train LRGR One-stage KD ====="
bash scripts/run_one_stage_senvtB_lrgr.sh

echo "===== Run Footprint for all models ====="
bash scripts/run_footprint_echo.sh

echo "===== Summarize Results ====="
bash scripts/run_summarize_echo.sh

echo "===== Summarize By Model ====="
bash scripts/run_summarize_by_model.sh

echo "===== DONE ====="