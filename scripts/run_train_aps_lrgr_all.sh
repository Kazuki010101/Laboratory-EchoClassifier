#!/bin/bash
set -e

echo "===== Footprint: APS-LRGR ====="
bash scripts/run_footprint_aps_lrgr.sh

echo "===== Train Baseline: APS-LRGR ====="
bash scripts/run_baseline_aps_lrgr.sh

echo "===== Train One-stage KD: SENvT-B -> APS-LRGR ====="
bash scripts/run_one_stage_senvtB_aps_lrgr.sh

echo "===== Summarize Results ====="
bash scripts/run_summarize_echo.sh

echo "===== Summarize By Model ====="
bash scripts/run_summarize_by_model.sh

echo "===== DONE ====="