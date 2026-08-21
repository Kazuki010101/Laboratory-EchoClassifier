#!/bin/bash
set -e

source scripts/config_stage1.sh

OUT_ROOT="$EXP_ROOT/one_stage_direct/senvtB_to_students"
mkdir -p "$OUT_ROOT"

echo "[One-stage] SENvT-B -> ResNet / DeepConvLSTM / MLPMixer"

for s in "${RESNET_STUDENTS[@]}" "${CNN_STUDENTS[@]}" "${MIXER_STUDENTS[@]}"; do
  OUT="$OUT_ROOT/senvtB_to_${s}"

  python main.py $COMMON_ARGS \
    --student "$s" \
    --model senvt-B \
    --teacher-path "$TEACHER_B" \
    $KD_ARGS \
    --output_dir "$OUT"
done

echo "[One-stage] SENvT-B -> PRC PatchEchoClassifier variants"

for cfg in "${ECHO_CONFIGS[@]}"; do
  read -r name student patch reservoir <<< "$cfg"

  OUT="$OUT_ROOT/senvtB_to_${name}"

  python main.py $COMMON_ARGS \
    --student "$student" \
    --patch_size "$patch" \
    --reservoir_size "$reservoir" \
    --model senvt-B \
    --teacher-path "$TEACHER_B" \
    $KD_ARGS \
    --output_dir "$OUT"
done

echo "[One-stage] SENvT-B -> PESAC variants"

for cfg in "${PESAC_CONFIGS[@]}"; do
  read -r name student patch reservoir <<< "$cfg"

  OUT="$OUT_ROOT/senvtB_to_${name}"

  python main.py $COMMON_ARGS \
    --student "$student" \
    --patch_size "$patch" \
    --reservoir_size "$reservoir" \
    --model senvt-B \
    --teacher-path "$TEACHER_B" \
    $KD_ARGS \
    --output_dir "$OUT"
done
