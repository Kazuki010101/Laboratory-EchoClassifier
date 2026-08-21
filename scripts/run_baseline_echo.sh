#!/bin/bash
set -e

source scripts/config_senvt_echo.sh

OUT_ROOT="$EXP_ROOT/stage0_baseline"
mkdir -p "$OUT_ROOT"

echo "[Baseline] ResNet / DeepConvLSTM / MLPMixer"

for s in "${RESNET_STUDENTS[@]}" "${CNN_STUDENTS[@]}" "${MIXER_STUDENTS[@]}"; do
  OUT="$OUT_ROOT/no_kd_to_${s}"

  python main.py $COMMON_ARGS \
    --student "$s" \
    --distillation-type none \
    --output_dir "$OUT"
done

echo "[Baseline] PRC PatchEchoClassifier variants"

for cfg in "${ECHO_CONFIGS[@]}"; do
  read -r name student patch reservoir <<< "$cfg"

  OUT="$OUT_ROOT/no_kd_to_${name}"

  python main.py $COMMON_ARGS \
    --student "$student" \
    --patch_size "$patch" \
    --reservoir_size "$reservoir" \
    --distillation-type none \
    --output_dir "$OUT"
done

echo "[Baseline] PESAC variants"

for cfg in "${PESAC_CONFIGS[@]}"; do
  read -r name student patch reservoir <<< "$cfg"

  OUT="$OUT_ROOT/no_kd_to_${name}"

  python main.py $COMMON_ARGS \
    --student "$student" \
    --patch_size "$patch" \
    --reservoir_size "$reservoir" \
    --distillation-type none \
    --output_dir "$OUT"
done
