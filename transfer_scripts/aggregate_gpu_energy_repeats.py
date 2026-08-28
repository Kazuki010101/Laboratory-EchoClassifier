#!/usr/bin/env python3
"""Aggregate repeated GPU energy measurements without mixing repeat and subject variance.

The input directory must contain files named
``transfer_gpu_energy_repeat01.csv`` through
``transfer_gpu_energy_repeat05.csv``.  Each detailed CSV is expected to contain
one row per model configuration and PAMAP2 test subject.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable


REPEAT_FILE_RE = re.compile(r"transfer_gpu_energy_repeat(\d{2})\.csv$")

METRICS = (
    "num_test_samples",
    "accuracy",
    "f1_macro",
    "precision_macro",
    "recall_macro",
    "params_total",
    "params_trainable",
    "parameter_bytes",
    "parameter_memory_mb",
    "model_size_mb",
    "num_patches",
    "selected_patches_mean",
    "actual_keep_ratio",
    "theoretical_reservoir_macs",
    "theoretical_total_macs",
    "thop_macs",
    "thop_flops_2x_macs",
    "thop_params",
    "routing_route_probability_mean",
    "routing_hard_keep_ratio",
    "routing_hard_keep_ratio_std",
    "routing_selected_patches_mean",
    "routing_innovation_distance_mean",
    "routing_innovation_cosine_mean",
    "latency_mean_ms",
    "latency_std_ms",
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_p99_ms",
    "throughput_samples_per_sec",
    "process_rss_before_mb",
    "process_rss_peak_mb",
    "process_rss_delta_peak_mb",
    "gpu_peak_allocated_mb",
    "gpu_peak_reserved_mb",
    "idle_power_w",
    "average_power_w",
    "gross_energy_j",
    "net_energy_j",
    "gross_joules_per_sample",
    "net_joules_per_sample",
    "samples_per_gross_joule",
    "correct_predictions_per_gross_joule",
)

REPEAT_VARIABILITY_METRICS = (
    "latency_mean_ms",
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_p99_ms",
    "throughput_samples_per_sec",
    "process_rss_before_mb",
    "process_rss_peak_mb",
    "process_rss_delta_peak_mb",
    "gpu_peak_allocated_mb",
    "gpu_peak_reserved_mb",
    "idle_power_w",
    "average_power_w",
    "gross_joules_per_sample",
    "net_joules_per_sample",
    "samples_per_gross_joule",
)

IDENTITY_COLUMNS = (
    "experiment_type",
    "model_key",
    "model",
    "dataset",
    "device",
    "test_subject",
    "val_subject",
    "patch_size",
    "reservoir_size",
    "reservoir_rank",
    "patch_keep_ratio",
    "innovation_target_ratio",
    "energy_backend",
)

MODEL_ORDER = {
    "PRC": 0,
    "PRC_LRGR": 1,
    "APS_LRGR": 2,
    "SIR_LRGR": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate repeat01--repeat05 GPU energy CSV files."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--expected-repeats", type=int, default=5)
    parser.add_argument("--expected-models", type=int, default=8)
    parser.add_argument("--expected-subjects", type=int, default=8)
    return parser.parse_args()


def as_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def mean(values: Iterable[float]) -> float | None:
    items = list(values)
    return statistics.fmean(items) if items else None


def sample_std(values: Iterable[float]) -> float | None:
    items = list(values)
    if not items:
        return None
    return statistics.stdev(items) if len(items) >= 2 else 0.0


def numeric_values(rows: Iterable[dict[str, object]], column: str) -> list[float]:
    result: list[float] = []
    for row in rows:
        value = as_float(row.get(column))
        if value is not None:
            result.append(value)
    return result


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def formatted(mean_value: object, std_value: object, scale: float = 1.0, digits: int = 2) -> str:
    m = as_float(mean_value)
    s = as_float(std_value)
    if m is None:
        return ""
    if s is None:
        return f"{m * scale:.{digits}f}"
    return f"{m * scale:.{digits}f} ({s * scale:.{digits}f})"


def model_sort_key(row: dict[str, object]) -> tuple[int, int, str]:
    patch = int(as_float(row.get("patch_size")) or 0)
    model = str(row.get("model", ""))
    return patch, MODEL_ORDER.get(model, 99), str(row.get("model_key", ""))


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or input_dir).resolve()

    repeat_files: list[tuple[int, Path]] = []
    for path in input_dir.glob("transfer_gpu_energy_repeat*.csv"):
        match = REPEAT_FILE_RE.fullmatch(path.name)
        if match:
            repeat_files.append((int(match.group(1)), path))
    repeat_files.sort()

    expected_ids = list(range(1, args.expected_repeats + 1))
    actual_ids = [repeat_id for repeat_id, _ in repeat_files]
    if actual_ids != expected_ids:
        raise RuntimeError(
            f"Expected repeat IDs {expected_ids}, but found {actual_ids} in {input_dir}"
        )

    all_rows: list[dict[str, object]] = []
    source_header: list[str] | None = None
    rows_per_file: dict[str, int] = {}
    for repeat_id, path in repeat_files:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            header = list(reader.fieldnames or [])
            if source_header is None:
                source_header = header
            elif header != source_header:
                raise RuntimeError(f"CSV header differs: {path}")
            rows = list(reader)
        rows_per_file[path.name] = len(rows)
        for row in rows:
            row["repeat_id"] = repeat_id
            all_rows.append(row)

    if not all_rows or source_header is None:
        raise RuntimeError("No detailed result rows were found.")

    required = {
        "experiment_type",
        "model_key",
        "model",
        "test_subject",
        "device",
        "energy_backend",
        "gross_joules_per_sample",
        "net_joules_per_sample",
    }
    missing = sorted(required.difference(source_header))
    if missing:
        raise RuntimeError(f"Required columns are missing: {missing}")

    model_keys = sorted({str(row["model_key"]) for row in all_rows})
    if len(model_keys) != args.expected_models:
        raise RuntimeError(
            f"Expected {args.expected_models} model configurations, found {len(model_keys)}: {model_keys}"
        )

    expected_rows_per_file = args.expected_models * args.expected_subjects
    bad_file_counts = {
        name: count for name, count in rows_per_file.items() if count != expected_rows_per_file
    }
    if bad_file_counts:
        raise RuntimeError(
            f"Each repeat must have {expected_rows_per_file} rows; mismatches: {bad_file_counts}"
        )

    non_transfer = sorted(
        {str(row["experiment_type"]) for row in all_rows if str(row["experiment_type"]) != "transfer"}
    )
    if non_transfer:
        raise RuntimeError(f"Non-transfer rows found: {non_transfer}")

    bad_energy_rows = [
        row
        for row in all_rows
        if str(row.get("energy_backend", "")).lower() != "nvml"
        or as_float(row.get("gross_joules_per_sample")) is None
        or as_float(row.get("net_joules_per_sample")) is None
    ]
    if bad_energy_rows:
        raise RuntimeError(
            f"{len(bad_energy_rows)} rows have a non-NVML backend or missing energy values."
        )

    subject_groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in all_rows:
        key = (str(row["model_key"]), str(row["test_subject"]), str(row["device"]))
        subject_groups[key].append(row)

    subject_rows: list[dict[str, object]] = []
    for key, rows in sorted(subject_groups.items()):
        repeat_ids = sorted(int(row["repeat_id"]) for row in rows)
        if repeat_ids != expected_ids:
            raise RuntimeError(f"Incomplete repeats for model/subject/device {key}: {repeat_ids}")
        first = rows[0]
        output: dict[str, object] = {
            column: first.get(column, "") for column in IDENTITY_COLUMNS
        }
        output["n_repeats"] = len(rows)
        output["repeat_ids"] = ";".join(f"{value:02d}" for value in repeat_ids)
        for metric in METRICS:
            values = numeric_values(rows, metric)
            output[metric] = mean(values)
            if metric in REPEAT_VARIABILITY_METRICS:
                output[f"{metric}_repeat_std"] = sample_std(values)
                metric_mean = mean(values)
                metric_std = sample_std(values)
                output[f"{metric}_repeat_cv_pct"] = (
                    100.0 * metric_std / abs(metric_mean)
                    if metric_mean not in (None, 0.0) and metric_std is not None
                    else None
                )
        subject_rows.append(output)

    subject_count_by_model: dict[str, set[str]] = defaultdict(set)
    for row in subject_rows:
        subject_count_by_model[str(row["model_key"])].add(str(row["test_subject"]))
    bad_subject_counts = {
        model: len(subjects)
        for model, subjects in subject_count_by_model.items()
        if len(subjects) != args.expected_subjects
    }
    if bad_subject_counts:
        raise RuntimeError(
            f"Each model must have {args.expected_subjects} test subjects; mismatches: {bad_subject_counts}"
        )

    model_groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in subject_rows:
        model_groups[(str(row["model_key"]), str(row["device"]))].append(row)

    final_rows: list[dict[str, object]] = []
    for _, rows in sorted(model_groups.items()):
        first = rows[0]
        output = {
            column: first.get(column, "")
            for column in IDENTITY_COLUMNS
            if column not in {"test_subject", "val_subject"}
        }
        output["n_subjects"] = len({str(row["test_subject"]) for row in rows})
        output["n_repeats"] = args.expected_repeats
        output["total_measurements"] = len(rows) * args.expected_repeats
        sample_counts = numeric_values(rows, "num_test_samples")
        accuracies = numeric_values(rows, "accuracy")
        output["num_test_samples_total"] = sum(sample_counts)
        if len(sample_counts) == len(accuracies) and sum(sample_counts) > 0:
            output["accuracy_sample_weighted"] = sum(
                count * accuracy for count, accuracy in zip(sample_counts, accuracies)
            ) / sum(sample_counts)
        else:
            output["accuracy_sample_weighted"] = None
        for metric in METRICS:
            values = numeric_values(rows, metric)
            output[f"{metric}_subject_mean"] = mean(values)
            output[f"{metric}_subject_std"] = sample_std(values)
            if metric in REPEAT_VARIABILITY_METRICS:
                repeat_stds = numeric_values(rows, f"{metric}_repeat_std")
                repeat_cvs = numeric_values(rows, f"{metric}_repeat_cv_pct")
                output[f"{metric}_repeat_std_mean"] = mean(repeat_stds)
                output[f"{metric}_repeat_cv_mean_pct"] = mean(repeat_cvs)
        final_rows.append(output)

    final_rows.sort(key=model_sort_key)

    paper_rows: list[dict[str, object]] = []
    for row in final_rows:
        model = str(row.get("model", ""))
        patch = int(as_float(row.get("patch_size")) or 0)
        paper_rows.append(
            {
                "Model": f"{model}-p{patch}",
                "Accuracy [%] mean (subject SD)": formatted(
                    row.get("accuracy_subject_mean"), row.get("accuracy_subject_std"), 100.0
                ),
                "Macro F1 [%] mean (subject SD)": formatted(
                    row.get("f1_macro_subject_mean"), row.get("f1_macro_subject_std"), 100.0
                ),
                "Model size [MB] mean (subject SD)": formatted(
                    row.get("model_size_mb_subject_mean"), row.get("model_size_mb_subject_std"), 1.0, 3
                ),
                "MACs [M] mean (subject SD)": formatted(
                    row.get("theoretical_total_macs_subject_mean"),
                    row.get("theoretical_total_macs_subject_std"),
                    1.0 / 1_000_000.0,
                    3,
                ),
                "GPU latency [ms] mean (subject SD)": formatted(
                    row.get("latency_mean_ms_subject_mean"), row.get("latency_mean_ms_subject_std"), 1.0, 3
                ),
                "Gross energy [mJ/sample] mean (subject SD)": formatted(
                    row.get("gross_joules_per_sample_subject_mean"),
                    row.get("gross_joules_per_sample_subject_std"),
                    1000.0,
                    3,
                ),
                "Net energy [mJ/sample] mean (subject SD)": formatted(
                    row.get("net_joules_per_sample_subject_mean"),
                    row.get("net_joules_per_sample_subject_std"),
                    1000.0,
                    3,
                ),
                "Actual keep [%] mean (subject SD)": formatted(
                    row.get("actual_keep_ratio_subject_mean"),
                    row.get("actual_keep_ratio_subject_std"),
                    100.0,
                ),
                "Gross energy repeat CV [%]": formatted(
                    row.get("gross_joules_per_sample_repeat_cv_mean_pct"), None, 1.0, 2
                ),
                "Latency repeat CV [%]": formatted(
                    row.get("latency_mean_ms_repeat_cv_mean_pct"), None, 1.0, 2
                ),
                "Subjects": row.get("n_subjects"),
                "Repeats per subject": row.get("n_repeats"),
            }
        )

    all_rows_path = output_dir / "gpu_energy_repeats_all_rows.csv"
    subject_path = output_dir / "gpu_energy_repeats_by_subject.csv"
    final_path = output_dir / "gpu_energy_repeats_final_summary.csv"
    paper_path = output_dir / "gpu_energy_repeats_paper_table.csv"
    validation_path = output_dir / "gpu_energy_repeats_validation.json"

    write_csv(all_rows_path, all_rows, ["repeat_id", *source_header])
    write_csv(subject_path, subject_rows, list(subject_rows[0].keys()))
    write_csv(final_path, final_rows, list(final_rows[0].keys()))
    write_csv(paper_path, paper_rows, list(paper_rows[0].keys()))

    validation = {
        "status": "ok",
        "input_directory": str(input_dir),
        "repeat_files": [path.name for _, path in repeat_files],
        "rows_per_repeat": rows_per_file,
        "detailed_rows": len(all_rows),
        "model_configurations": len(model_keys),
        "subjects_per_model": args.expected_subjects,
        "repeats_per_subject": args.expected_repeats,
        "subject_averaged_rows": len(subject_rows),
        "final_summary_rows": len(final_rows),
        "energy_backend": "nvml",
        "aggregation_order": [
            "mean over five repeated measurements for each model and test subject",
            "mean and sample standard deviation over test subjects for each model",
        ],
        "outputs": [
            path.name
            for path in (all_rows_path, subject_path, final_path, paper_path)
        ],
    }
    validation_path.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("[ok] repeated GPU energy results were aggregated")
    print(f"[ok] detailed rows: {len(all_rows)}")
    print(f"[ok] subject-averaged rows: {len(subject_rows)}")
    print(f"[ok] final model rows: {len(final_rows)}")
    for path in (subject_path, final_path, paper_path, validation_path):
        print(f"[output] {path}")


if __name__ == "__main__":
    main()
