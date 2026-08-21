#!/bin/bash
set -e

source scripts/config_senvt_echo.sh

OUT_ROOT="$EXP_ROOT/reference_paper_style/mlpmixer_to_students"
mkdir -p "$OUT_ROOT"

echo "[Reference] 1DMLP-Mixer -> PRC PatchEchoClassifier variants"

for cfg in "${ECHO_CONFIGS[@]}"; do
  read -r name student patch reservoir <<< "$cfg"

  OUT="$OUT_ROOT/mlpmixer_to_${name}"

  python main.py $COMMON_ARGS \
    --student "$student" \
    --patch_size "$patch" \
    --reservoir_size "$reservoir" \
    --model MLPMixer \
    $KD_ARGS \
    --output_dir "$OUT"
done

echo "[Reference] 1DMLP-Mixer -> PESAC variants"

for cfg in "${PESAC_CONFIGS[@]}"; do
  read -r name student patch reservoir <<< "$cfg"

  OUT="$OUT_ROOT/mlpmixer_to_${name}"

  python main.py $COMMON_ARGS \
    --student "$student" \
    --patch_size "$patch" \
    --reservoir_size "$reservoir" \
    --model MLPMixer \
    $KD_ARGS \
    --output_dir "$OUT"
done

echo "[Reference] 1DMLP-Mixer -> MLPMixer student"

python main.py $COMMON_ARGS \
  --student MLPMixer \
  --model MLPMixer \
  $KD_ARGS \
  --output_dir "$OUT_ROOT/mlpmixer_to_MLPMixer"

echo "[Reference] 1DMLP-Mixer -> ResNet / DeepConvLSTM"

for s in "${RESNET_STUDENTS[@]}" "${CNN_STUDENTS[@]}"; do
  OUT="$OUT_ROOT/mlpmixer_to_${s}"

  python main.py $COMMON_ARGS \
    --student "$s" \
    --model MLPMixer \
    $KD_ARGS \
    --output_dir "$OUT"
done
