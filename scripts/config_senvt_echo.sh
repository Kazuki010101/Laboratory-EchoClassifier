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
# format: name student patch_size reservoir_size
PESAC_CONFIGS=(
  "PESAC_p32_r1000 PESAC 32 1000"
  "PESAC_p64_r1000 PESAC 64 1000"
  "PESAC_p128_r1000 PESAC 128 1000"
  "PESAC_p128_r4000 PESAC 128 4000"
)

# Low-Rank Gated Reservoir variants
# format: name student patch_size reservoir_size reservoir_rank
LRGR_CONFIGS=(
  "LRGR_p32_r1000_rank32 PRC_LRGR 32 1000 32"
  "LRGR_p64_r1000_rank32 PRC_LRGR 64 1000 32"
  "LRGR_p128_r1000_rank32 PRC_LRGR 128 1000 32"
  "LRGR_p128_r1000_rank64 PRC_LRGR 128 1000 64"
  "LRGR_p128_r4000_rank64 PRC_LRGR 128 4000 64"
)

# Adaptive Patch-Skipping Low-Rank Gated Reservoir variants
# format: name student patch_size reservoir_size reservoir_rank patch_keep_ratio
APS_LRGR_CONFIGS=(
  "APS_LRGR_p32_r1000_rank32_keep050 APS_LRGR 32 1000 32 0.50"
  "APS_LRGR_p64_r1000_rank32_keep050 APS_LRGR 64 1000 32 0.50"
  "APS_LRGR_p128_r1000_rank32_keep067 APS_LRGR 128 1000 32 0.67"
  "APS_LRGR_p128_r4000_rank64_keep067 APS_LRGR 128 4000 64 0.67"
)

# Input-Factorized State-Attentive Low-Rank Gated Reservoir variants
# format: name student patch_size reservoir_size reservoir_rank input_rank

# State Innovation Routed LRGR
#
# format:
# name
# student
# patch_size
# reservoir_size
# reservoir_rank
# target_keep_ratio
# routing_threshold
SIR_LRGR_CONFIGS=(

  # p32 / reservoir 1000
  # "SIR_LRGR_p32_r1000_rank32_target050 SIR_LRGR 32 1000 32 0.50 0.50"
  # "SIR_LRGR_p32_r1000_rank64_target050 SIR_LRGR 32 1000 64 0.50 0.50"

  # # p64 / reservoir 1000
  # "SIR_LRGR_p64_r1000_rank32_target050 SIR_LRGR 64 1000 32 0.50 0.50"
  # "SIR_LRGR_p64_r1000_rank64_target050 SIR_LRGR 64 1000 64 0.50 0.50"

  # # p128 / reservoir 1000
  # "SIR_LRGR_p128_r1000_rank32_target067 SIR_LRGR 128 1000 32 0.67 0.50"
  "SIR_LRGR_p128_r1000_rank64_target067 SIR_LRGR 128 1000 64 0.67 0.50"
  "SIR_LRGR_p128_r1000_rank128_target067 SIR_LRGR 128 1000 128 0.67 0.50"

  # p128 / reservoir 4000
  "SIR_LRGR_p128_r4000_rank64_target067 SIR_LRGR 128 4000 64 0.67 0.50"
  "SIR_LRGR_p128_r4000_rank128_target067 SIR_LRGR 128 4000 128 0.67 0.50"
)


IFSA_LRGR_CONFIGS=(
  "IFSA_LRGR_p64_r1000_rank32_irank16 IFSA_LRGR 64 1000 32 16"
  "IFSA_LRGR_p128_r1000_rank32_irank16 IFSA_LRGR 128 1000 32 16"
  "IFSA_LRGR_p128_r1000_rank32_irank32 IFSA_LRGR 128 1000 32 32"
  "IFSA_LRGR_p128_r4000_rank64_irank32 IFSA_LRGR 128 4000 64 32"
)