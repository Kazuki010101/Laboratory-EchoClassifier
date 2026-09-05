# SHL-2023 post-standard-KD scripts

Copy all files in this directory to:

```text
scripts/shl2023_senvt_kd/
```

Keep the existing `config.sh` in that directory. Then apply the two small code
changes in `USER23_DATASET_PATCH.md`.

Run syntax checks:

```bash
chmod +x scripts/shl2023_senvt_kd/*.sh
bash -n scripts/shl2023_senvt_kd/11_prepare_shl2023_user23.sh
bash -n scripts/shl2023_senvt_kd/12_eval_checkpoint_user23.sh
bash -n scripts/shl2023_senvt_kd/13_eval_standard_kd_user23.sh
bash -n scripts/shl2023_senvt_kd/14_train_prc_ce.sh
bash -n scripts/shl2023_senvt_kd/15_eval_prc_ce_user23.sh
bash -n scripts/shl2023_senvt_kd/19_after_10_distill.sh
python -m py_compile scripts/shl2023_senvt_kd/prepare_shl2023_user23.py
```

Immediately after `10_distill_standard_students.sh` finishes:

```bash
bash scripts/shl2023_senvt_kd/19_after_10_distill.sh
```

This prepares the official mixed User2/User3 validation arrays and evaluates
only the standard-KD PRC, which is the primary control for the proposed PRC
method.

The preparer intentionally reproduces the existing `raw2npy.py` behavior,
including its median window label and exact `np.split` boundary rule. Do not
run `raw2npy.py --datafiles ...` directly: it currently calls `parse_args([])`,
so command-line arguments are ignored.

To evaluate every already-completed comparison model later:

```bash
bash scripts/shl2023_senvt_kd/13_eval_standard_kd_user23.sh all_completed
```

Train and evaluate the non-distilled PRC control:

```bash
bash scripts/shl2023_senvt_kd/14_train_prc_ce.sh smoke
bash scripts/shl2023_senvt_kd/14_train_prc_ce.sh full
bash scripts/shl2023_senvt_kd/15_eval_prc_ce_user23.sh
```

Do not use the challenge `test` directory for local accuracy. It contains
timestamps (`Label_idx.txt`) rather than ground-truth activity labels.
