# Required dataset support before running cross-user evaluation

The official SHL-2023 validation data is a **mixture of User2 and User3**.
The distributed files do not provide a per-window user ID, so the defensible
metric is the mixed User2/User3 validation score, not two separate scores.

Before running `12_eval_checkpoint_user23.sh`, make these two small changes.

## `main.py`

Add `SHL2023_user23` to the choices of the existing `--data` argument.

Also add this guard near the beginning of `main(args)`, before datasets are
built. It prevents accidental training on the held-out split.

```python
if args.data == "SHL2023_user23" and not args.eval:
    raise RuntimeError(
        "SHL2023_user23 is evaluation-only. Add --eval."
    )
```

## `datasets.py`

Insert this branch immediately before the existing `elif args.data ==
"SHL2023_test":` branch:

```python
elif args.data == "SHL2023_user23":
    acc_path = os.path.join(args.data_path, "Hips_Acc.npy")
    label_path = os.path.join(args.data_path, "Hips_Label.npy")

    acc_xyz = np.load(acc_path, mmap_mode="r")
    label = np.load(label_path, mmap_mode="r")
    label = np.asarray(label, dtype=np.int64) - 1

    if acc_xyz.ndim != 3 or acc_xyz.shape[1:] != (3, 500):
        raise RuntimeError(f"Unexpected User2/3 Acc shape: {acc_xyz.shape}")
    if label.ndim != 1 or len(acc_xyz) != len(label):
        raise RuntimeError("User2/3 Acc/Label count mismatch")
    if label.min() != 0 or label.max() != 7:
        raise RuntimeError("Expected User2/3 model labels 0..7")

    eval_data = TriaxialSignalDataset(
        data=acc_xyz,
        labels=label,
        transform=build_transform(is_train=False, args=args),
    )

    # main.py currently expects both train and val objects even in --eval mode.
    # The main.py guard above guarantees this split cannot be trained on.
    return eval_data, eval_data, 8
```

Do not route this branch into the common `train_test_split` block. Every
User2/User3 validation window must remain held out and be evaluated exactly
once.

