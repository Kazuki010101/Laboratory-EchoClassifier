#!/usr/bin/env python3
"""Create the paper-compatible fixed 80/10/10 User2/User3 split."""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    output = Path(args.output)
    if output.exists() and not args.force:
        raise SystemExit(f"[STOP] Split already exists: {output} (use --force to replace)")

    labels = np.load(args.label, mmap_mode="r")
    indices = np.arange(len(labels), dtype=np.int64)
    train_indices, remaining = train_test_split(
        indices, train_size=0.8, random_state=args.seed, shuffle=True)
    val_indices, test_indices = train_test_split(
        remaining, train_size=0.5, random_state=args.seed, shuffle=True)
    train_indices = np.sort(train_indices)
    val_indices = np.sort(val_indices)
    test_indices = np.sort(test_indices)

    merged = np.concatenate([train_indices, val_indices, test_indices])
    if len(np.unique(merged)) != len(labels) or len(merged) != len(labels):
        raise RuntimeError("Split overlap or omission detected")
    expected = (22985, 2873, 2874)
    actual = (len(train_indices), len(val_indices), len(test_indices))
    if len(labels) == 28732 and actual != expected:
        raise RuntimeError(f"Expected {expected}, got {actual}")

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, train_indices=train_indices, val_indices=val_indices,
             test_indices=test_indices, seed=np.int64(args.seed))
    report = {
        "label_file": str(Path(args.label).resolve()), "total": len(labels),
        "seed": args.seed, "train_count": actual[0], "val_count": actual[1],
        "test_count": actual[2], "overlap_count": 0,
        "label_values": np.unique(labels).astype(int).tolist(),
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"[OK] {output}")


if __name__ == "__main__":
    main()

