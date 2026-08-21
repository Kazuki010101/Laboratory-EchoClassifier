#!/bin/bash

source scripts/config_senvt_echo.sh

PAMAP2_ROOT="dataset/pamap2_100hz_w496_s248_chest16g"

PAMAP2_BASELINE_EXP_ROOT="$EXP_ROOT/pamap2_baseline_all_models"
PAMAP2_TRANSFER_EXP_ROOT="$EXP_ROOT/pamap2_transfer_all_models"

PAMAP2_TEST_SUBJECTS="101 102 103 104 105 106 107 108"
PAMAP2_DROP_SUBJ=109

PAMAP2_COMMON_ARGS="--data PAMAP2 \
  --pamap2_root $PAMAP2_ROOT \
  --pamap2_drop_subj $PAMAP2_DROP_SUBJ \
  --input-size 496 \
  --epochs 100 \
  --batch-size 64 \
  --mixup 0 \
  --cutmix 0 \
  --clip-grad 1.0 \
  --lr 5e-4 \
  --warmup-epochs 2 \
  --distillation-type none \
  --device cuda"

PAMAP2_ALL_MODEL_CONFIGS=(

  # "MLPMixer MLPMixer 16 0 0 1.00"

  # "Resnet_S Resnet_S 16 0 0 1.00"
  # "Resnet_M Resnet_M 16 0 0 1.00"
  # "Resnet_L Resnet_L 16 0 0 1.00"

  # "DeepConvLSTM25 DeepConvLSTM25 16 0 0 1.00"
  # "DeepConvLSTM50 DeepConvLSTM50 16 0 0 1.00"
  # "DeepConvLSTM100 DeepConvLSTM100 16 0 0 1.00"

  # "PRC_p32_r1000 PRC 32 1000 0 1.00"
  # "PRC_p64_r1000 PRC 64 1000 0 1.00"
  # "PRC_p128_r1000 PRC 128 1000 0 1.00"
  # "PRC_p128_r4000 PRC 128 4000 0 1.00"

  # "PESAC_p32_r1000 PESAC 32 1000 0 1.00"
  # "PESAC_p64_r1000 PESAC 64 1000 0 1.00"
  # "PESAC_p128_r1000 PESAC 128 1000 0 1.00"
  # "PESAC_p128_r4000 PESAC 128 4000 0 1.00"

  # "LRGR_p32_r1000_rank32 PRC_LRGR 32 1000 32 1.00"
  # "LRGR_p64_r1000_rank32 PRC_LRGR 64 1000 32 1.00"
  # "LRGR_p128_r1000_rank32 PRC_LRGR 128 1000 32 1.00"
  # "LRGR_p128_r1000_rank64 PRC_LRGR 128 1000 64 1.00"
  # "LRGR_p128_r4000_rank64 PRC_LRGR 128 4000 64 1.00"

  # "APS_LRGR_p32_r1000_rank32_keep050 APS_LRGR 32 1000 32 0.50"
  # "APS_LRGR_p64_r1000_rank32_keep050 APS_LRGR 64 1000 32 0.50"
  # "APS_LRGR_p128_r1000_rank32_keep067 APS_LRGR 128 1000 32 0.67"
  # "APS_LRGR_p128_r4000_rank64_keep067 APS_LRGR 128 4000 64 0.67"
  
  "IFSA_LRGR_p64_r1000_rank32_irank16 IFSA_LRGR 64 1000 32 1.00 16"
  "IFSA_LRGR_p128_r1000_rank32_irank16 IFSA_LRGR 128 1000 32 1.00 16"
  "IFSA_LRGR_p128_r1000_rank32_irank32 IFSA_LRGR 128 1000 32 1.00 32"
  "IFSA_LRGR_p128_r4000_rank64_irank32 IFSA_LRGR 128 4000 64 1.00 32"
)

PAMAP2_SIR_MODEL_CONFIGS=(

  "SIR_LRGR_p32_r1000_rank32_target050 SIR_LRGR 32 1000 32 0.50 0.50"
  "SIR_LRGR_p32_r1000_rank64_target050 SIR_LRGR 32 1000 64 0.50 0.50"

  "SIR_LRGR_p64_r1000_rank32_target050 SIR_LRGR 64 1000 32 0.50 0.50"
  "SIR_LRGR_p64_r1000_rank64_target050 SIR_LRGR 64 1000 64 0.50 0.50"

  "SIR_LRGR_p128_r1000_rank32_target067 SIR_LRGR 128 1000 32 0.67 0.50"
  "SIR_LRGR_p128_r1000_rank64_target067 SIR_LRGR 128 1000 64 0.67 0.50"
  "SIR_LRGR_p128_r1000_rank128_target067 SIR_LRGR 128 1000 128 0.67 0.50"

  "SIR_LRGR_p128_r4000_rank64_target067 SIR_LRGR 128 4000 64 0.67 0.50"
  "SIR_LRGR_p128_r4000_rank128_target067 SIR_LRGR 128 4000 128 0.67 0.50"
)

get_senvtb_transfer_checkpoint() {
  local name="$1"
  echo "$EXP_ROOT/one_stage_direct/senvtB_to_students/senvtB_to_${name}/best_checkpoint.pth"
}