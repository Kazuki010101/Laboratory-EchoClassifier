#!/usr/bin/env python3
"""Fill energy columns in an efficiency summary from a separate energy run."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


KEY_FIELDS = ("experiment_type", "model_key", "device")
ENERGY_FIELDS = (
    "energy_backend",
    "idle_power_w_subject_mean",
    "idle_power_w_subject_std",
    "average_power_w_subject_mean",
    "average_power_w_subject_std",
    "gross_energy_j_subject_mean",
    "gross_energy_j_subject_std",
    "net_energy_j_subject_mean",
    "net_energy_j_subject_std",
    "gross_joules_per_sample_subject_mean",
    "gross_joules_per_sample_subject_std",
    "net_joules_per_sample_subject_mean",
    "net_joules_per_sample_subject_std",
    "samples_per_gross_joule_subject_mean",
    "samples_per_gross_joule_subject_std",
    "correct_predictions_per_gross_joule_subject_mean",
    "correct_predictions_per_gross_joule_subject_std",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--energy", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def row_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(str(row.get(field, "")).strip() for field in KEY_FIELDS)


def main() -> None:
    args = parse_args()
    base_fields, base_rows = read_csv(args.base)
    energy_fields, energy_rows = read_csv(args.energy)

    missing_base_fields = [field for field in KEY_FIELDS + ENERGY_FIELDS if field not in base_fields]
    if missing_base_fields:
        raise RuntimeError(f"Base summary lacks columns: {missing_base_fields}")

    missing_energy_fields = [field for field in KEY_FIELDS + ENERGY_FIELDS if field not in energy_fields]
    if missing_energy_fields:
        raise RuntimeError(f"Energy summary lacks columns: {missing_energy_fields}")

    lookup: dict[tuple[str, ...], dict[str, str]] = {}
    for row in energy_rows:
        key = row_key(row)
        if key in lookup:
            raise RuntimeError(f"Duplicate energy-summary key: {key}")
        if not str(row.get("energy_backend", "")).strip():
            raise RuntimeError(
                f"Energy backend is blank for {key}. RAPL measurement did not succeed."
            )
        if not str(row.get("gross_joules_per_sample_subject_mean", "")).strip():
            raise RuntimeError(f"Gross energy is blank for {key}")
        lookup[key] = row

    matched = 0
    missing_keys: list[tuple[str, ...]] = []
    for row in base_rows:
        key = row_key(row)
        energy_row = lookup.get(key)
        if energy_row is None:
            missing_keys.append(key)
            continue
        for field in ENERGY_FIELDS:
            row[field] = energy_row[field]
        matched += 1

    if missing_keys:
        preview = ", ".join(str(key) for key in missing_keys[:5])
        raise RuntimeError(
            f"Energy summary is missing {len(missing_keys)} base rows. First keys: {preview}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=base_fields)
        writer.writeheader()
        writer.writerows(base_rows)

    print(f"[done] merged rows: {matched}")
    print(f"[done] output: {args.output}")


if __name__ == "__main__":
    main()
