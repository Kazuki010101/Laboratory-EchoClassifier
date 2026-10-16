#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


LABELS = np.array([1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24], dtype=np.int64)
SUBJECTS = (101, 102, 103, 104, 105, 106, 107, 108)
VALIDATION_SUBJECT = {
    101: 102,
    102: 103,
    103: 104,
    104: 105,
    105: 106,
    106: 107,
    107: 108,
    108: 101,
}


def class_counts(labels):
    values, counts = np.unique(labels, return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(values, counts)}


def main():
    parser = argparse.ArgumentParser(
        description="Create fixed subject-disjoint PAMAP2 train/validation/test folds."
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--drop-subject", type=int, default=109)
    parser.add_argument("--input-size", type=int, default=496)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    x = np.load(root / "X.npy", mmap_mode="r")
    y = np.asarray(np.load(root / "Y.npy", mmap_mode="r"))
    groups = np.asarray(np.load(root / "groups.npy", mmap_mode="r"))

    if x.shape[1:] != (3, args.input_size):
        raise ValueError(f"Expected X shape [N, 3, {args.input_size}], got {x.shape}")
    if len(x) != len(y) or len(y) != len(groups):
        raise ValueError(f"Sample counts differ: X={len(x)}, Y={len(y)}, groups={len(groups)}")
    if not np.isfinite(y).all() or not np.isfinite(groups).all():
        raise ValueError("Y or groups contains NaN/Inf")
    if set(np.unique(y).tolist()) != set(LABELS.tolist()):
        raise ValueError(
            f"Expected labels {LABELS.tolist()}, got {np.unique(y).tolist()}"
        )

    arrays = {}
    manifest = {
        "dataset_root": str(root),
        "input_shape": list(x.shape),
        "raw_labels": LABELS.tolist(),
        "label_to_index": {str(int(label)): index for index, label in enumerate(LABELS)},
        "included_subjects": list(SUBJECTS),
        "dropped_subject": args.drop_subject,
        "split_policy": "outer test subject plus one cyclic validation subject",
        "folds": {},
    }

    for test_subject in SUBJECTS:
        val_subject = VALIDATION_SUBJECT[test_subject]
        train_subjects = [
            subject for subject in SUBJECTS
            if subject not in {test_subject, val_subject}
        ]
        train_idx = np.flatnonzero(np.isin(groups, train_subjects)).astype(np.int64)
        val_idx = np.flatnonzero(groups == val_subject).astype(np.int64)
        test_idx = np.flatnonzero(groups == test_subject).astype(np.int64)

        if not len(train_idx) or not len(val_idx) or not len(test_idx):
            raise ValueError(f"Fold {test_subject} contains an empty split")
        if np.intersect1d(train_idx, val_idx).size:
            raise ValueError(f"Fold {test_subject}: train/validation overlap")
        if np.intersect1d(train_idx, test_idx).size:
            raise ValueError(f"Fold {test_subject}: train/test overlap")
        if np.intersect1d(val_idx, test_idx).size:
            raise ValueError(f"Fold {test_subject}: validation/test overlap")
        if set(np.unique(y[train_idx]).tolist()) != set(LABELS.tolist()):
            raise ValueError(f"Fold {test_subject}: training split lacks a class")
        used_groups = set(groups[np.concatenate([train_idx, val_idx, test_idx])].tolist())
        if args.drop_subject in used_groups:
            raise ValueError(f"Fold {test_subject}: dropped subject is present")

        prefix = f"fold_{test_subject}"
        arrays[f"{prefix}_train_indices"] = train_idx
        arrays[f"{prefix}_val_indices"] = val_idx
        arrays[f"{prefix}_test_indices"] = test_idx
        manifest["folds"][str(test_subject)] = {
            "train_subjects": train_subjects,
            "validation_subject": val_subject,
            "test_subject": test_subject,
            "train_samples": len(train_idx),
            "validation_samples": len(val_idx),
            "test_samples": len(test_idx),
            "train_class_counts": class_counts(y[train_idx]),
            "validation_class_counts": class_counts(y[val_idx]),
            "test_class_counts": class_counts(y[test_idx]),
        }

    output = Path(args.output).resolve()
    manifest_path = Path(args.manifest).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[OK] {output}")
    print(f"[OK] {manifest_path}")
    for fold, info in manifest["folds"].items():
        print(
            f"fold={fold} train={info['train_samples']} "
            f"val={info['validation_samples']} test={info['test_samples']}"
        )


if __name__ == "__main__":
    main()

