#!/bin/bash

EXP_ROOT=experiments/SHL2023_senvt_echo_w496
PRETRAINED_SENVT_B=dataset/SENvT-u4/1000k_task4/best.pth

# SHL2023でfine-tuningしたSENvT-B。蒸留teacherとして使う。
TEACHER_B=$EXP_ROOT/teacher_finetune/senvtB/best_checkpoint.pth

COMMON_ARGS="--data SHL2023 --input-size 496 \
             --epochs 100 --batch-size 128 \
             --mixup 0 --cutmix 0 --clip-grad 1.0"

KD_ARGS="--distillation-type soft \
         --distillation-alpha 0.7 \
         --distillation-tau 2.5"

# ResNet系 student
RESNET_STUDENTS=(
  Resnet_S
  Resnet_M
  Resnet_L
)

# DeepConvLSTM系 student
CNN_STUDENTS=(
  DeepConvLSTM25
  DeepConvLSTM50
  DeepConvLSTM100
)

# PatchMixerClassifier相当
MIXER_STUDENTS=(
  MLPMixer
)

# 論文の PatchEchoClassifier に対応する設定
# format: name student patch_size reservoir_size
ECHO_CONFIGS=(
  "PRC_p32_r1000 PRC 32 1000"
  "PRC_p64_r1000 PRC 64 1000"
  "PRC_p128_r1000 PRC 128 1000"
  "PRC_p128_r4000 PRC 128 4000"
)

# PatchEchoClassifierの発展版として比較するPESAC
PESAC_CONFIGS=(
  "PESAC_p32_r1000 PESAC 32 1000"
  "PESAC_p64_r1000 PESAC 64 1000"
  "PESAC_p128_r1000 PESAC 128 1000"
  "PESAC_p128_r4000 PESAC 128 4000"
)
