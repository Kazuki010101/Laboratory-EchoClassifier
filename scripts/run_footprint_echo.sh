#!/bin/bash
set -eo pipefail

source scripts/config_senvt_echo.sh

CSV="$EXP_ROOT/footprint/model_static_metrics.csv"
mkdir -p "$EXP_ROOT/footprint"

echo "========================================"
echo "[Footprint] output csv: $CSV"
echo "========================================"

echo "[Footprint] ResNet / DeepConvLSTM / MLPMixer"

for s in "${RESNET_STUDENTS[@]}" "${CNN_STUDENTS[@]}" "${MIXER_STUDENTS[@]}"; do
  echo "----------------------------------------"
  echo "[Footprint] $s"
  echo "----------------------------------------"

  python model4edgetest.py \
    --dataset SHL2023 \
    --run-name "$s" \
    --model "$s" \
    --input-size 496 \
    --num-classes 8 \
    --batch-size 1 \
    --device cpu \
    --csv-path "$CSV"
done

echo "[Footprint] PRC PatchEchoClassifier variants"

for cfg in "${ECHO_CONFIGS[@]}"; do
  read -r name student patch reservoir <<< "$cfg"

  echo "----------------------------------------"
  echo "[Footprint] $name"
  echo "----------------------------------------"

  python model4edgetest.py \
    --dataset SHL2023 \
    --run-name "$name" \
    --model "$student" \
    --patch_size "$patch" \
    --reservoir_size "$reservoir" \
    --input-size 496 \
    --num-classes 8 \
    --batch-size 1 \
    --device cpu \
    --csv-path "$CSV"
done

echo "[Footprint] PESAC variants"

for cfg in "${PESAC_CONFIGS[@]}"; do
  read -r name student patch reservoir <<< "$cfg"

  echo "----------------------------------------"
  echo "[Footprint] $name"
  echo "----------------------------------------"

  python model4edgetest.py \
    --dataset SHL2023 \
    --run-name "$name" \
    --model "$student" \
    --patch_size "$patch" \
    --reservoir_size "$reservoir" \
    --input-size 496 \
    --num-classes 8 \
    --batch-size 1 \
    --device cpu \
    --csv-path "$CSV"
done

echo "[Footprint] LRGR variants"

for cfg in "${LRGR_CONFIGS[@]}"; do
  read -r name student patch reservoir rank <<< "$cfg"

  echo "----------------------------------------"
  echo "[Footprint] $name"
  echo "----------------------------------------"

  python model4edgetest.py \
    --dataset SHL2023 \
    --run-name "$name" \
    --model "$student" \
    --patch_size "$patch" \
    --reservoir_size "$reservoir" \
    --reservoir_rank "$rank" \
    --input-size 496 \
    --num-classes 8 \
    --batch-size 1 \
    --device cpu \
    --csv-path "$CSV"
done

echo "========================================"
echo "[done] Footprint CSV generated: $CSV"
echo "========================================"