#!/bin/bash
set -e

bash scripts/run_footprint_echo.sh
bash scripts/run_baseline_echo.sh
bash scripts/run_stage1_senvt.sh
bash scripts/run_one_stage_senvtB_echo.sh
bash scripts/run_stage2_senvtXS_echo.sh
bash scripts/run_stage2_senvtS_echo.sh

# 論文再現・参考比較として必要な場合
bash scripts/run_reference_mlpmixer_echo.sh
