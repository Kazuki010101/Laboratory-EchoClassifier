#!/usr/bin/env python3
"""Prepare SHL-2023 validation exactly like the existing raw2npy.py.

Compatibility details reproduced here:
* detect timestamp gaps above 10 ms;
* use np.split(..., gap_indexes) without adding one;
* create non-overlapping 500-sample windows;
* use the median window label cast to int64;
* save float16 acceleration with shape [N, 3, 500].
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acc", required=True, type=Path)
    parser.add_argument("--label", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--window-size", default=500, type=int)
    parser.add_argument("--expected-step-ms", default=10, type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_acc = args.output_dir / "Hips_Acc.npy"
    output_label = args.output_dir / "Hips_Label.npy"

    for path in (args.acc, args.label):
        if not path.is_file():
            raise FileNotFoundError(path)

    if not args.force and (output_acc.exists() or output_label.exists()):
        raise RuntimeError(
            "Output already exists. Inspect it first or rerun with --force: "
            f"{args.output_dir}"
        )

    print("[1/5] Loading validation Acc.txt...", flush=True)
    acc_raw = np.loadtxt(args.acc)
    print("[2/5] Loading validation Label.txt...", flush=True)
    label_raw = np.loadtxt(args.label)

    if acc_raw.ndim != 2 or acc_raw.shape[1] != 4:
        raise RuntimeError(f"Unexpected Acc shape: {acc_raw.shape}")
    if label_raw.ndim != 2 or label_raw.shape[1] != 2:
        raise RuntimeError(f"Unexpected Label shape: {label_raw.shape}")
    if len(acc_raw) != len(label_raw):
        raise RuntimeError("Acc.txt and Label.txt have different line counts")
    if not np.array_equal(acc_raw[:, 0], label_raw[:, 0]):
        raise RuntimeError("Acc/Label timestamps are not aligned")

    print("[3/5] Reproducing raw2npy.py timestamp splits...", flush=True)
    gap_indexes = np.flatnonzero(
        np.diff(acc_raw[:, 0]) > args.expected_step_ms
    )

    # Deliberately do not add one: this matches ProcessSHL.data_load().
    acc_splits = np.split(acc_raw, gap_indexes)
    label_splits = np.split(label_raw, gap_indexes)

    print("[4/5] Creating compatible 500-sample windows...", flush=True)
    windows: list[np.ndarray] = []
    labels: list[int] = []

    for acc_split, label_split in zip(acc_splits, label_splits):
        values = acc_split[:, 1:4]
        label_values = label_split[:, 1]

        # Identical to time_range(0, len(index)-1, step=500).
        for start in range(0, len(acc_split) - 1, args.window_size):
            window = values[start : start + args.window_size]
            label_window = label_values[start : start + args.window_size]
            if len(window) != args.window_size:
                continue
            if len(label_window) != len(window):
                raise RuntimeError("Window-level Acc/Label mismatch")

            windows.append(window.T.astype(np.float16, copy=False))
            labels.append(int(np.median(label_window)))

    if not windows:
        raise RuntimeError("No complete windows were generated")

    acc_array = np.stack(windows, axis=0).astype(np.float16, copy=False)
    label_array = np.asarray(labels, dtype=np.int64)
    if len(acc_array) != len(label_array):
        raise AssertionError("Acc/Label window count mismatch")

    values, counts = np.unique(label_array, return_counts=True)
    if not np.array_equal(values, np.arange(1, 9)):
        raise RuntimeError(f"Expected labels 1..8, found {values.tolist()}")

    print("[5/5] Saving arrays atomically...", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporary_acc = output_acc.with_suffix(".npy.tmp")
    temporary_label = output_label.with_suffix(".npy.tmp")
    with temporary_acc.open("wb") as file:
        np.save(file, acc_array)
    with temporary_label.open("wb") as file:
        np.save(file, label_array)
    temporary_acc.replace(output_acc)
    temporary_label.replace(output_label)

    print("=== SHL-2023 USER2/3 VALIDATION PREPARATION ===")
    print("compatibility: raw2npy.py ProcessSHL")
    print("raw_rows:", len(acc_raw))
    print("gap_count:", len(gap_indexes))
    print("acc_shape:", acc_array.shape)
    print("acc_dtype:", acc_array.dtype)
    print("label_shape:", label_array.shape)
    print("label_dtype:", label_array.dtype)
    print("labels:", dict(zip(values.tolist(), counts.tolist())))
    print("saved_acc:", output_acc)
    print("saved_label:", output_label)


if __name__ == "__main__":
    main()
