#!/bin/bash
set -e

echo "===== Train IFSA-LRGR Baseline ====="
bash scripts/run_baseline_ifsa_lrgr.sh

echo "===== Train IFSA-LRGR One-stage KD ====="
bash scripts/run_one_stage_senvtB_ifsa_lrgr.sh

echo "===== Footprint IFSA-LRGR ====="
bash scripts/run_footprint_ifsa_lrgr.sh

echo "===== DONE IFSA-LRGR All Training ====="