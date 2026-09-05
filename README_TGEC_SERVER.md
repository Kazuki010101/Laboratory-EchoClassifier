# TGEC server integration for SHL-2023

This ZIP is an **overlay** for the existing EchoClassifier repository. Copy its
contents into the repository root. It retains standard KD and adds a controlled
full-rank PRC study.

## What is compared

| Run | 31 patches | Student routing | Omitted patches | Teacher losses |
|---|---:|---|---|---|
| existing `PRC` | all processed | none | none | logit KD |
| `aps_ce` | top 16 | task-trained APS | discarded | none |
| `aps_kd` | top 16 | task-trained APS | discarded | logit KD |
| `tg_skip` | top 16 | SENvT layer-1 evidence norm | discarded | logit + route |
| `tgec` | top 16 + one summary | SENvT layer-1 evidence norm | mean/RMS summary token | logit + route + content |

All routed variants use the original **full-rank PRC reservoir**. LRGR and SIR
are not part of the proposed-method comparison. Evidence norm is primary because
the 40-sample diagnostic found Attention alone unreliable while layer-1 Value
evidence remained temporally discriminative.

The total TGEC objective is:

`(1-alpha) CE + alpha logit_KD + route_weight route_KL + content_weight cosine_loss`.

Content loss aligns the summary token with a deterministic projection of the
teacher's omitted layer-1 CLS Value contributions. The teacher is used only for
training; inference runs the student alone.

## Install and verify

```bash
cd /home/jovyan/work/srv11/蒸留/EchoClassifier
python -m py_compile main.py engine.py loss_func.py datasets.py \
  models/TeacherGuidedEvidenceCondensation.py models/senvt_adapters.py
bash -n scripts/shl2023_senvt_kd/*.sh
python scripts/shl2023_senvt_kd/00_smoke_test_tgec.py
```

## Recommended execution order

First run one epoch for every new condition:

```bash
bash scripts/shl2023_senvt_kd/20_run_prc_condensation_study.sh smoke all
```

If all four complete, run the full study. Completed outputs are skipped; partial
outputs stop safely unless resume is explicitly requested.

```bash
bash scripts/shl2023_senvt_kd/20_run_prc_condensation_study.sh full all
# after an interruption:
RESUME_PARTIAL=1 bash scripts/shl2023_senvt_kd/20_run_prc_condensation_study.sh full all
```

Prepare the official labeled validation mixture (User2/User3) once, then evaluate:

```bash
bash scripts/shl2023_senvt_kd/11_prepare_shl2023_user23.sh
bash scripts/shl2023_senvt_kd/21_eval_prc_study_user23.sh
```

The official validation set is a User2/User3 mixture without per-window user IDs,
so this reports held-out mixed-user accuracy, not separate User2 and User3 scores.

Finally profile deploy-time latency, peak CUDA memory and dominant reservoir MACs:

```bash
bash scripts/shl2023_senvt_kd/22_profile_prc_study.sh
```

Results are written below `experiments/shl2023_senvt_kd/`. Do not delete the
existing standard-KD outputs; the PRC standard-KD checkpoint is the full-patch
reference condition.
