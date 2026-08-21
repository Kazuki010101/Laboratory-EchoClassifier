#!/bin/bash

# project root から実行する前提
source scripts/config_senvt_echo.sh

PAMAP2_ROOT="dataset/pamap2_100hz_w496_s248_chest16g"
PAMAP2_EXP_ROOT="$EXP_ROOT/pamap2_transfer"

# PAMAP2 subject split
# 109はdrop対象として扱う
PAMAP2_TEST_SUBJECTS="101 102 103 104 105 106 107 108"
PAMAP2_DROP_SUBJ=109

# format:
# name student patch_size reservoir_size reservoir_rank patch_keep_ratio transfer_checkpoint
#
# 注意:
# - MLPMixer / ResNet / DeepConvLSTM は patch_size, reservoir_size, reservoir_rank, patch_keep_ratio を実質ダミーとして入れている
# - PRC / PESAC は reservoir_rank を 0 にしているだけ
# - PRC / PESAC は patch_keep_ratio を 1.00 にしているだけ
# - LRGR は patch_keep_ratio 1.00
# - APS_LRGR は実験時の keep ratio を入れる
#
# 今回は、完了済みモデルは消さずにコメントアウトし、
# まだPAMAP2 transfer testができていないモデルだけを有効化する。

PAMAP2_TRANSFER_CONFIGS=(

  # =========================
  # MLPMixer: 未実行なので有効
  # =========================
  # "MLPMixer MLPMixer 16 0 0 1.00 $EXP_ROOT/one_stage_direct/senvtB_to_students/senvtB_to_MLPMixer/best_checkpoint.pth"

  # =========================
  # ResNet family: 未実行なので有効
  # =========================
  # "Resnet_S Resnet_S 16 0 0 1.00 $EXP_ROOT/one_stage_direct/senvtB_to_students/senvtB_to_Resnet_S/best_checkpoint.pth"
  # "Resnet_M Resnet_M 16 0 0 1.00 $EXP_ROOT/one_stage_direct/senvtB_to_students/senvtB_to_Resnet_M/best_checkpoint.pth"
  # "Resnet_L Resnet_L 16 0 0 1.00 $EXP_ROOT/one_stage_direct/senvtB_to_students/senvtB_to_Resnet_L/best_checkpoint.pth"

  # # =========================
  # DeepConvLSTM family: 未実行なので有効
  # # =========================
  # "DeepConvLSTM25 DeepConvLSTM25 16 0 0 1.00 $EXP_ROOT/one_stage_direct/senvtB_to_students/senvtB_to_DeepConvLSTM25/best_checkpoint.pth"
  # "DeepConvLSTM50 DeepConvLSTM50 16 0 0 1.00 $EXP_ROOT/one_stage_direct/senvtB_to_students/senvtB_to_DeepConvLSTM50/best_checkpoint.pth"
  # "DeepConvLSTM100 DeepConvLSTM100 16 0 0 1.00 $EXP_ROOT/one_stage_direct/senvtB_to_students/senvtB_to_DeepConvLSTM100/best_checkpoint.pth"


  # =========================
  # PRC: PatchEchoClassifier
  # 完了済みのためコメントアウト
  # =========================
  # "PRC_p32_r1000 PRC 32 1000 0 1.00 $EXP_ROOT/one_stage_direct/senvtB_to_students/senvtB_to_PRC_p32_r1000/best_checkpoint.pth"
  # "PRC_p64_r1000 PRC 64 1000 0 1.00 $EXP_ROOT/one_stage_direct/senvtB_to_students/senvtB_to_PRC_p64_r1000/best_checkpoint.pth"
  # "PRC_p128_r1000 PRC 128 1000 0 1.00 $EXP_ROOT/one_stage_direct/senvtB_to_students/senvtB_to_PRC_p128_r1000/best_checkpoint.pth"
  # "PRC_p128_r4000 PRC 128 4000 0 1.00 $EXP_ROOT/one_stage_direct/senvtB_to_students/senvtB_to_PRC_p128_r4000/best_checkpoint.pth"


  # =========================
  # PESAC: Attention系PRC
  # 完了済みのためコメントアウト
  # =========================
  # "PESAC_p32_r1000 PESAC 32 1000 0 1.00 $EXP_ROOT/one_stage_direct/senvtB_to_students/senvtB_to_PESAC_p32_r1000/best_checkpoint.pth"
  # "PESAC_p64_r1000 PESAC 64 1000 0 1.00 $EXP_ROOT/one_stage_direct/senvtB_to_students/senvtB_to_PESAC_p64_r1000/best_checkpoint.pth"
  # "PESAC_p128_r1000 PESAC 128 1000 0 1.00 $EXP_ROOT/one_stage_direct/senvtB_to_students/senvtB_to_PESAC_p128_r1000/best_checkpoint.pth"
  # "PESAC_p128_r4000 PESAC 128 4000 0 1.00 $EXP_ROOT/one_stage_direct/senvtB_to_students/senvtB_to_PESAC_p128_r4000/best_checkpoint.pth"


  # =========================
  # LRGR: Low-Rank Gated Reservoir
  # 完了済みのためコメントアウト
  # =========================
  # "LRGR_p32_r1000_rank32 PRC_LRGR 32 1000 32 1.00 $EXP_ROOT/one_stage_direct/senvtB_to_students/senvtB_to_LRGR_p32_r1000_rank32/best_checkpoint.pth"
  # "LRGR_p64_r1000_rank32 PRC_LRGR 64 1000 32 1.00 $EXP_ROOT/one_stage_direct/senvtB_to_students/senvtB_to_LRGR_p64_r1000_rank32/best_checkpoint.pth"
  # "LRGR_p128_r1000_rank32 PRC_LRGR 128 1000 32 1.00 $EXP_ROOT/one_stage_direct/senvtB_to_students/senvtB_to_LRGR_p128_r1000_rank32/best_checkpoint.pth"
  # "LRGR_p128_r1000_rank64 PRC_LRGR 128 1000 64 1.00 $EXP_ROOT/one_stage_direct/senvtB_to_students/senvtB_to_LRGR_p128_r1000_rank64/best_checkpoint.pth"
  # "LRGR_p128_r4000_rank64 PRC_LRGR 128 4000 64 1.00 $EXP_ROOT/one_stage_direct/senvtB_to_students/senvtB_to_LRGR_p128_r4000_rank64/best_checkpoint.pth"


  # =========================
  # APS-LRGR: Adaptive Patch-Skipping LRGR
  # p32は完了済み想定なのでコメントアウト
  # p64以降は未完了なので有効
  # =========================
  # "APS_LRGR_p32_r1000_rank32_keep050 APS_LRGR 32 1000 32 0.50 $EXP_ROOT/one_stage_direct/senvtB_to_students/senvtB_to_APS_LRGR_p32_r1000_rank32_keep050/best_checkpoint.pth"

  # "APS_LRGR_p64_r1000_rank32_keep050 APS_LRGR 64 1000 32 0.50 $EXP_ROOT/one_stage_direct/senvtB_to_students/senvtB_to_APS_LRGR_p64_r1000_rank32_keep050/best_checkpoint.pth"
  # "APS_LRGR_p128_r1000_rank32_keep067 APS_LRGR 128 1000 32 0.67 $EXP_ROOT/one_stage_direct/senvtB_to_students/senvtB_to_APS_LRGR_p128_r1000_rank32_keep067/best_checkpoint.pth"
  # "APS_LRGR_p128_r4000_rank64_keep067 APS_LRGR 128 4000 64 0.67 $EXP_ROOT/one_stage_direct/senvtB_to_students/senvtB_to_APS_LRGR_p128_r4000_rank64_keep067/best_checkpoint.pth"

    # =========================
  # IFSA-LRGR: Input-Factorized State-Attentive LRGR
  # =========================
  # "IFSA_LRGR_p64_r1000_rank32_irank16 IFSA_LRGR 64 1000 32 1.00 16 $EXP_ROOT/one_stage_direct/senvtB_to_students/senvtB_to_IFSA_LRGR_p64_r1000_rank32_irank16/best_checkpoint.pth"
  # "IFSA_LRGR_p128_r1000_rank32_irank16 IFSA_LRGR 128 1000 32 1.00 16 $EXP_ROOT/one_stage_direct/senvtB_to_students/senvtB_to_IFSA_LRGR_p128_r1000_rank32_irank16/best_checkpoint.pth"
  # "IFSA_LRGR_p128_r1000_rank32_irank32 IFSA_LRGR 128 1000 32 1.00 32 $EXP_ROOT/one_stage_direct/senvtB_to_students/senvtB_to_IFSA_LRGR_p128_r1000_rank32_irank32/best_checkpoint.pth"
  # "IFSA_LRGR_p128_r4000_rank64_irank32 IFSA_LRGR 128 4000 64 1.00 32 $EXP_ROOT/one_stage_direct/senvtB_to_students/senvtB_to_IFSA_LRGR_p128_r4000_rank64_irank32/best_checkpoint.pth"
)

PAMAP2_SIR_TRANSFER_CONFIGS=(
  "SIR_LRGR_p32_r1000_rank32_target050 SIR_LRGR 32 1000 32 0.50 0.50"
  "SIR_LRGR_p64_r1000_rank32_target050 SIR_LRGR 64 1000 32 0.50 0.50"
  "SIR_LRGR_p128_r1000_rank32_target067 SIR_LRGR 128 1000 32 0.67 0.50"
)