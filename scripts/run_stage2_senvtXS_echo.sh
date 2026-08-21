#!/bin/bash
set -e

source scripts/config_senvt_echo.sh

TEACHER_XS="$EXP_ROOT/stage1_intermediate_teacher/senvtB_to_senvtXS/best_checkpoint.pth"

OUT_ROOT="$EXP_ROOT/stage2_two_stage/senvtXS_to_students"
mkdir -p "$OUT_ROOT"

echo "[Two-stage XS] SENvT-B -> SENvT-XS -> ResNet / DeepConvLSTM / MLPMixer"

for s in "${RESNET_STUDENTS[@]}" "${CNN_STUDENTS[@]}" "${MIXER_STUDENTS[@]}"; do
  OUT="$OUT_ROOT/senvtXS_to_${s}"

  python main.py $COMMON_ARGS \
    --student "$s" \
    --model senvt-XS \
    --teacher-path "$TEACHER_XS" \
    $KD_ARGS \
    --output_dir "$OUT"
done

echo "[Two-stage XS] SENvT-B -> SENvT-XS -> PRC PatchEchoClassifier variants"

for cfg in "${ECHO_CONFIGS[@]}"; do
  read -r name student patch reservoir <<< "$cfg"

  OUT="$OUT_ROOT/senvtXS_to_${name}"

  python main.py $COMMON_ARGS \
    --student "$student" \
    --patch_size "$patch" \
    --reservoir_size "$reservoir" \
    --model senvt-XS \
    --teacher-path "$TEACHER_XS" \
    $KD_ARGS \
    --output_dir "$OUT"
done

echo "[Two-stage XS] SENvT-B -> SENvT-XS -> PESAC variants"

for cfg in "${PESAC_CONFIGS[@]}"; do
  read -r name student patch reservoir <<< "$cfg"

  OUT="$OUT_ROOT/senvtXS_to_${name}"

  python main.py $COMMON_ARGS \
    --student "$student" \
    --patch_size "$patch" \
    --reservoir_size "$reservoir" \
    --model senvt-XS \
    --teacher-path "$TEACHER_XS" \
    $KD_ARGS \
    --output_dir "$OUT"
done
