#!/usr/bin/env python3
"""Aggregate evaluation.json files across experiment seeds.

The experiment directory is expected to contain folders such as::

    aps_kd_seed0/evaluation.json
    aps_kd_seed1/evaluation.json
    PRC_kd_seed0/evaluation.json
    prc_kd_seed1/evaluation.json

Folder-name matching is case-insensitive.  By default, only seed 0 and seed 1
are aggregated.  The script uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


RUN_DIR_RE = re.compile(r"^(?P<method>.+)_seed(?P<seed>\d+)$", re.IGNORECASE)

DISPLAY_NAMES = {
    "prc_ce": "PRC-CE",
    "prc_kd": "PRC-KD",
    "aps_ce": "APS-CE",
    "aps_kd": "APS-KD",
    "tg_skip": "TG-Skip",
    "tgec": "TGEC",
}

# The first existing alias is used. Nested keys are matched by their final key.
METRIC_ALIASES = {
    "loss": ("loss", "test_loss", "eval_loss"),
    "accuracy": ("accuracy", "acc1", "test_acc1", "eval_acc1", "top1"),
    "acc5": ("acc5", "test_acc5", "eval_acc5", "top5"),
    "precision_macro": (
        "precision_macro",
        "macro_precision",
        "test_precision_macro",
        "eval_precision_macro",
    ),
    "recall_macro": (
        "recall_macro",
        "macro_recall",
        "test_recall_macro",
        "eval_recall_macro",
    ),
    "f1_macro": (
        "f1_macro",
        "macro_f1",
        "test_f1_macro",
        "eval_f1_macro",
    ),
    "routing_patch_count": ("routing_patch_count", "test_routing_patch_count"),
    "routing_keep_count": ("routing_keep_count", "test_routing_keep_count"),
    "routing_keep_ratio": ("routing_keep_ratio", "test_routing_keep_ratio"),
    "routing_route_entropy": (
        "routing_route_entropy",
        "route_entropy",
        "test_routing_route_entropy",
    ),
    "n_parameters": ("n_parameters", "parameters", "parameter_count"),
    "flops": ("flops", "flops_total", "total_flops"),
    "macs": ("macs", "macs_total", "total_macs", "estimated_dominant_macs"),
    "latency_ms": ("latency_ms", "median_latency_ms", "latency_median_ms"),
    "throughput_samples_s": (
        "throughput_samples_s",
        "throughput_samples_per_second",
        "throughput",
    ),
    "peak_memory_mb": ("peak_memory_mb", "max_memory_mb", "max_memory_mb_all"),
    "footprint_mb": ("footprint_mb", "checkpoint_size_mb", "model_size_mb"),
}

PERCENT_METRICS = {
    "accuracy",
    "acc5",
    "precision_macro",
    "recall_macro",
    "f1_macro",
}

PREFERRED_MARKDOWN_METRICS = (
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "loss",
    "routing_patch_count",
    "routing_keep_count",
    "routing_keep_ratio",
    "routing_route_entropy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recursively aggregate evaluation_summary.json for selected seeds."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Directory containing *_seedN folders (default: current directory).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1],
        help="Seeds to include (default: 0 1).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: ROOT/aggregate_seed0_seed1).",
    )
    parser.add_argument(
        "--population-std",
        action="store_true",
        help="Use population SD (ddof=0). Default is sample SD (ddof=1).",
    )
    return parser.parse_args()


def canonical_method(raw: str) -> str:
    method = re.sub(r"[-\s]+", "_", raw.strip().lower())
    method = re.sub(r"_+", "_", method).strip("_")
    return method


def display_method(method: str) -> str:
    return DISPLAY_NAMES.get(method, method.replace("_", "-").upper())


def run_info(path: Path, root: Path) -> tuple[str, int] | None:
    """Find the nearest ancestor named METHOD_seedN."""
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        return None

    for part in reversed(relative.parts[:-1]):
        match = RUN_DIR_RE.match(part)
        if match:
            return canonical_method(match.group("method")), int(match.group("seed"))
    return None


def flatten_numeric(value: Any, prefix: str = "") -> dict[str, float]:
    """Flatten finite numeric scalars from a JSON object."""
    result: dict[str, float] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten_numeric(child, child_prefix))
    elif isinstance(value, list):
        # Per-class arrays and confusion matrices are intentionally not averaged.
        return result
    elif isinstance(value, bool):
        return result
    elif isinstance(value, (int, float)) and math.isfinite(float(value)):
        result[prefix] = float(value)
    return result


def normalized_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")


def find_metric(flat: dict[str, float], aliases: Iterable[str]) -> float | None:
    normalized = [(key, normalized_key(key), value) for key, value in flat.items()]
    for alias in aliases:
        alias_norm = normalized_key(alias)
        # Exact full-key match has priority.
        for _, key_norm, value in normalized:
            if key_norm == alias_norm:
                return value
        # Then accept a nested leaf such as metrics.accuracy.
        for key, _, value in normalized:
            if normalized_key(key.split(".")[-1]) == alias_norm:
                return value
    return None


def as_percent(value: float) -> float:
    """Accept both 0..1 and 0..100 conventions and return percent."""
    return value * 100.0 if abs(value) <= 1.0 else value


def load_record(path: Path, root: Path, method: str, seed: int) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("top-level JSON must be an object")

    flat = flatten_numeric(payload)
    record: dict[str, Any] = {
        "method": method,
        "method_label": display_method(method),
        "seed": seed,
        "evaluation_file": str(path.resolve().relative_to(root.resolve())),
    }
    for metric, aliases in METRIC_ALIASES.items():
        value = find_metric(flat, aliases)
        if value is not None:
            record[metric] = as_percent(value) if metric in PERCENT_METRICS else value

    # Preserve other numeric scalars for routing/KD diagnostics without allowing
    # them to overwrite canonical columns above.
    for key, value in flat.items():
        extra_key = f"json.{normalized_key(key)}"
        if extra_key not in record:
            record[extra_key] = value
    return record


def finite_numbers(records: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for record in records:
        value = record.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = float(value)
            if math.isfinite(value):
                values.append(value)
    return values


def summarize(
    records: list[dict[str, Any]], population_std: bool
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["method"])].append(record)

    metric_keys = sorted(
        {
            key
            for record in records
            for key, value in record.items()
            if key not in {"method", "method_label", "seed", "evaluation_file"}
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        }
    )
    rows: list[dict[str, Any]] = []
    for method in sorted(grouped, key=lambda name: display_method(name).lower()):
        method_records = sorted(grouped[method], key=lambda row: int(row["seed"]))
        row: dict[str, Any] = {
            "method": method,
            "method_label": display_method(method),
            "seeds": ",".join(str(int(record["seed"])) for record in method_records),
            "n_seeds": len({int(record["seed"]) for record in method_records}),
        }
        for key in metric_keys:
            values = finite_numbers(method_records, key)
            if not values:
                continue
            mean = statistics.fmean(values)
            if len(values) >= 2:
                sd = statistics.pstdev(values) if population_std else statistics.stdev(values)
                formatted = f"{mean:.4f} ± {sd:.4f}"
            else:
                sd = None
                formatted = f"{mean:.4f} ± N/A"
            row[f"{key}_mean"] = mean
            row[f"{key}_std"] = sd
            row[f"{key}_mean_std"] = formatted
        rows.append(row)
    return rows


def ordered_columns(rows: list[dict[str, Any]], first: list[str]) -> list[str]:
    keys = {key for row in rows for key in row}
    return [key for key in first if key in keys] + sorted(keys.difference(first))


def write_csv(path: Path, rows: list[dict[str, Any]], first: list[str]) -> None:
    columns = ordered_columns(rows, first)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def markdown_metric_name(metric: str) -> str:
    return {
        "accuracy": "Accuracy [%]",
        "precision_macro": "Macro Precision [%]",
        "recall_macro": "Macro Recall [%]",
        "f1_macro": "Macro F1 [%]",
    }.get(metric, metric)


def write_markdown(path: Path, summaries: list[dict[str, Any]]) -> None:
    available = [
        metric
        for metric in PREFERRED_MARKDOWN_METRICS
        if any(f"{metric}_mean_std" in row for row in summaries)
    ]
    headers = ["Method", "Seeds", *[markdown_metric_name(m) for m in available]]
    lines = [
        "# Seed evaluation summary",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + " --- |" * len(headers),
    ]
    for row in summaries:
        cells = [str(row["method_label"]), str(row["seeds"])]
        for metric in available:
            cells.append(str(row.get(f"{metric}_mean_std", "N/A")))
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "- Classification metrics are expressed as percentages.",
            "- Standard deviation is calculated across seeds; N/A means only one seed was found.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        print(f"ERROR: root directory does not exist: {root}", file=sys.stderr)
        return 2

    selected_seeds = set(args.seeds)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else root / ("aggregate_seed" + "_seed".join(map(str, args.seeds)))
    )

    candidates: list[tuple[Path, str, int]] = []
    for path in sorted(root.rglob("evaluation_summary.json")):
        info = run_info(path, root)
        if info is None:
            print(f"WARNING: skipped (no *_seedN ancestor): {path}", file=sys.stderr)
            continue
        method, seed = info
        if seed in selected_seeds:
            candidates.append((path, method, seed))

    if not candidates:
        print(
            f"ERROR: no evaluation.json found for seeds {sorted(selected_seeds)} under {root}",
            file=sys.stderr,
        )
        return 1

    seen_runs: dict[tuple[str, int], Path] = {}
    records: list[dict[str, Any]] = []
    for path, method, seed in candidates:
        run_key = (method, seed)
        if run_key in seen_runs:
            print(
                "ERROR: multiple evaluation.json files found for "
                f"{display_method(method)} seed{seed}:\n"
                f"  {seen_runs[run_key]}\n  {path}",
                file=sys.stderr,
            )
            return 1
        seen_runs[run_key] = path
        try:
            records.append(load_record(path, root, method, seed))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            print(f"ERROR: failed to read {path}: {exc}", file=sys.stderr)
            return 1

    records.sort(key=lambda row: (str(row["method_label"]).lower(), int(row["seed"])))
    summaries = summarize(records, args.population_std)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(
        output_dir / "evaluation_by_seed.csv",
        records,
        ["method", "method_label", "seed", "evaluation_file"],
    )
    write_csv(
        output_dir / "evaluation_mean_std.csv",
        summaries,
        ["method", "method_label", "seeds", "n_seeds"],
    )
    write_markdown(output_dir / "evaluation_mean_std.md", summaries)
    with (output_dir / "evaluation_aggregate.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "root": str(root),
                "selected_seeds": sorted(selected_seeds),
                "std": "population" if args.population_std else "sample",
                "records": records,
                "summary": summaries,
            },
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")

    found_by_method = defaultdict(set)
    for record in records:
        found_by_method[str(record["method_label"])].add(int(record["seed"]))
    for method_label, found in sorted(found_by_method.items()):
        missing = selected_seeds.difference(found)
        if missing:
            print(
                f"WARNING: {method_label} is missing seed(s): "
                + ", ".join(map(str, sorted(missing))),
                file=sys.stderr,
            )

    print(f"Found {len(records)} evaluation files.")
    print(f"Saved aggregate results to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
