import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


TAG_PATTERN = re.compile(
    r"^(prc_ce|prc_kd|aps_ce|aps_kd|tg_skip|tgec)"
    r"(?:_ps(?P<patch_size>\d+)_k(?P<keep_count>\d+))?"
    r"_seed(?P<seed>\d+)(?P<smoke>_smoke)?$",
    re.IGNORECASE,
)

METHOD_NAMES = {
    "prc_ce": "PRC-CE",
    "prc_kd": "PRC-KD",
    "aps_ce": "APS-CE",
    "aps_kd": "APS-KD",
    "tg_skip": "TG-Skip",
    "tgec": "TGEC",
}

EFFICIENCY_KEYS = {
    "prc_ce": "prc_full",
    "prc_kd": "prc_full",
    "aps_ce": "aps_prc",
    "aps_kd": "aps_prc",
    "tg_skip": "tg_skip_prc",
    "tgec": "tgec_prc",
}


def read_json(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def first(mapping, *keys):
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def first_existing(candidates):
    for path in candidates:
        if path.is_file():
            return path
    return None


def as_percent(value):
    if value is None:
        return None
    value = float(value)
    return value * 100.0 if abs(value) <= 1.0 else value


def upstream_summary_path(root, method, tag, seed, smoke):
    suffix = "_smoke" if smoke else ""

    if method == "prc_ce":
        candidates = [
            root / "student_baseline" / tag / "summary.json",
            root
            / "student_baseline"
            / f"PRC_ce_seed{seed}{suffix}"
            / "summary.json",
            root
            / "student_baseline"
            / f"prc_ce_seed{seed}{suffix}"
            / "summary.json",
        ]
    elif method == "prc_kd":
        candidates = [
            root / "student_distillation" / "standard_kd" / tag / "summary.json",
            root
            / "student_distillation"
            / "standard_kd"
            / f"PRC_seed{seed}{suffix}"
            / "summary.json",
            root
            / "student_distillation"
            / "standard_kd"
            / f"prc_kd_seed{seed}{suffix}"
            / "summary.json",
        ]
    else:
        candidates = [
            root
            / "student_distillation"
            / "prc_condensation"
            / tag
            / "summary.json"
        ]

    return first_existing(candidates)


def finetune_summary_path(root, tag, method, seed, smoke):
    suffix = "_smoke" if smoke else ""
    candidates = [
        root / "student_finetune" / "user23" / tag / "summary.json",
        root
        / "student_finetune"
        / "user23"
        / f"{method}_seed{seed}{suffix}"
        / "summary.json",
    ]

    if method == "prc_ce":
        candidates.append(
            root
            / "student_finetune"
            / "user23"
            / f"PRC_ce_seed{seed}{suffix}"
            / "summary.json"
        )
    elif method == "prc_kd":
        candidates.append(
            root
            / "student_finetune"
            / "user23"
            / f"PRC_kd_seed{seed}{suffix}"
            / "summary.json"
        )

    return first_existing(candidates)


def efficiency_candidates(root, patch_size, keep_count, device):
    results = root / "results"
    candidates = [
        results / f"prc_efficiency_ps{patch_size}_k{keep_count}_{device}.json",
    ]

    if device == "gpu":
        candidates.append(
            results / f"prc_efficiency_ps{patch_size}_k{keep_count}.json"
        )
        if patch_size == 16 and keep_count == 16:
            candidates.extend(
                [
                    results / "prc_efficiency_ps16_k16.json",
                    results / "prc_efficiency.json",
                ]
            )

    return candidates


def load_efficiency(root, patch_size, keep_count, device):
    path = first_existing(
        efficiency_candidates(root, patch_size, keep_count, device)
    )
    return (read_json(path), path) if path else ({}, None)


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_sd(values):
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None, None
    return (
        statistics.mean(clean),
        statistics.stdev(clean) if len(clean) >= 2 else None,
    )


def fmt(mean, sd):
    if mean is None:
        return "-"
    if sd is None:
        return f"{mean:.3f}"
    return f"{mean:.3f} ± {sd:.3f}"


def fmt_number(value, digits=3):
    return "-" if value is None else f"{float(value):.{digits}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-root",
        default="experiments/shl2023_senvt_kd",
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    root = Path(args.experiment_root).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else root / "results"
    )
    evaluation_root = root / "heldout_evaluation" / "user23_test"

    rows = []
    for evaluation_path in sorted(
        evaluation_root.glob("*/evaluation_summary.json")
    ):
        tag = evaluation_path.parent.name
        match = TAG_PATTERN.match(tag)
        if not match:
            print(f"[WARN] Unrecognized result directory: {tag}")
            continue

        method = match.group(1).lower()
        seed = int(match.group("seed"))
        smoke = bool(match.group("smoke"))
        patch_size = int(match.group("patch_size") or 16)

        keep_count_group = match.group("keep_count")
        if keep_count_group is None:
            keep_count = 31 if method.startswith("prc_") else 16
        else:
            keep_count = int(keep_count_group)

        evaluation = read_json(evaluation_path)

        upstream_path = upstream_summary_path(
            root,
            method,
            tag,
            seed,
            smoke,
        )
        finetune_path = finetune_summary_path(
            root,
            tag,
            method,
            seed,
            smoke,
        )
        upstream = read_json(upstream_path) if upstream_path else {}
        finetune = read_json(finetune_path) if finetune_path else {}

        # PRC uses all 31 patches, but its efficiency data is stored in the
        # common k16 profile together with routed models.
        efficiency_keep_count = 16 if method.startswith("prc_") else keep_count

        gpu_json, gpu_path = load_efficiency(
            root,
            patch_size,
            efficiency_keep_count,
            "gpu",
        )
        cpu_json, cpu_path = load_efficiency(
            root,
            patch_size,
            efficiency_keep_count,
            "cpu",
        )

        efficiency_key = EFFICIENCY_KEYS[method]
        gpu_efficiency = gpu_json.get(efficiency_key, {})
        cpu_efficiency = cpu_json.get(efficiency_key, {})
        static_efficiency = gpu_efficiency or cpu_efficiency

        measured_keep_count = first(static_efficiency, "keep_count")
        if measured_keep_count is None:
            measured_keep_count = keep_count

        measured_keep_ratio = first(static_efficiency, "keep_ratio")
        if measured_keep_ratio is None:
            measured_keep_ratio = measured_keep_count / 31

        dominant_macs = first(
            static_efficiency,
            "estimated_dominant_macs_per_sample",
            "estimated_dominant_macs",
        )
        dominant_mmacs = first(
            static_efficiency,
            "estimated_dominant_mmacs_per_sample",
        )
        if dominant_mmacs is None and dominant_macs is not None:
            dominant_mmacs = float(dominant_macs) / 1e6

        profiled_mflops = first(
            static_efficiency,
            "profiled_mflops_per_sample",
            "profiled_mflops",
        )

        row = {
            "method": METHOD_NAMES[method],
            "method_key": method,
            "tag": tag,
            "seed": seed,
            "patch_size": patch_size,
            "keep_count": measured_keep_count,
            "keep_ratio": measured_keep_ratio,
            "user1_val_accuracy": first(upstream, "best_acc1"),
            "user23_val_accuracy": first(finetune, "best_acc1"),
            "test_accuracy_percent": as_percent(first(evaluation, "acc1")),
            "test_precision_macro_percent": as_percent(
                first(evaluation, "precision_macro")
            ),
            "test_recall_macro_percent": as_percent(
                first(evaluation, "recall_macro")
            ),
            "test_f1_macro_percent": as_percent(
                first(evaluation, "f1_macro")
            ),
            "test_loss": first(evaluation, "loss"),
            "test_samples": first(evaluation, "num_samples"),
            "route_entropy": first(
                evaluation,
                "routing_route_entropy",
                "route_entropy",
            ),
            "trainable_parameters": first(
                static_efficiency,
                "trainable_parameters",
            ),
            "frozen_parameters": first(
                static_efficiency,
                "frozen_parameters",
            ),
            "buffer_elements": first(
                static_efficiency,
                "buffer_elements",
            ),
            "model_state_elements": first(
                static_efficiency,
                "model_state_elements",
                "parameters",
            ),
            "model_state_size_mib": first(
                static_efficiency,
                "model_state_size_mb",
            ),
            "profiled_mflops_per_sample": profiled_mflops,
            "estimated_dominant_mmacs_per_sample": dominant_mmacs,
            "recurrent_tokens_including_cls_dist": first(
                static_efficiency,
                "recurrent_tokens_including_cls_dist",
            ),
            "gpu_latency_ms_median": first(
                gpu_efficiency,
                "latency_ms_median",
            ),
            "gpu_latency_ms_mean": first(
                gpu_efficiency,
                "latency_ms_mean",
            ),
            "gpu_latency_ms_stdev": first(
                gpu_efficiency,
                "latency_ms_stdev",
            ),
            "gpu_throughput_samples_per_sec": first(
                gpu_efficiency,
                "throughput_samples_per_sec",
            ),
            "gpu_peak_memory_mib": first(
                gpu_efficiency,
                "peak_memory_mb",
            ),
            "cpu_latency_ms_median": first(
                cpu_efficiency,
                "latency_ms_median",
            ),
            "cpu_latency_ms_mean": first(
                cpu_efficiency,
                "latency_ms_mean",
            ),
            "cpu_latency_ms_stdev": first(
                cpu_efficiency,
                "latency_ms_stdev",
            ),
            "cpu_throughput_samples_per_sec": first(
                cpu_efficiency,
                "throughput_samples_per_sec",
            ),
            "efficiency_source_gpu": str(gpu_path) if gpu_path else "",
            "efficiency_source_cpu": str(cpu_path) if cpu_path else "",
        }
        rows.append(row)

    if not rows:
        raise SystemExit(
            f"No evaluation_summary.json found under {evaluation_root}"
        )

    detailed_fields = list(rows[0].keys())
    detailed_path = output_dir / "prc_combined_results.csv"
    write_csv(detailed_path, rows, detailed_fields)

    groups = defaultdict(list)
    for row in rows:
        group_key = (
            row["method"],
            row["patch_size"],
            int(float(row["keep_count"])),
        )
        groups[group_key].append(row)

    summary_rows = []
    for (method, patch_size, keep_count), group in sorted(groups.items()):
        summary = {
            "method": method,
            "patch_size": patch_size,
            "keep_count": keep_count,
            "n_seeds": len({row["seed"] for row in group}),
            "seeds": " ".join(
                str(seed) for seed in sorted({row["seed"] for row in group})
            ),
        }

        for key in [
            "test_accuracy_percent",
            "test_precision_macro_percent",
            "test_recall_macro_percent",
            "test_f1_macro_percent",
        ]:
            mean, sd = mean_sd([row[key] for row in group])
            summary[f"{key}_mean"] = mean
            summary[f"{key}_sd"] = sd

        first_row = group[0]
        for key in [
            "trainable_parameters",
            "frozen_parameters",
            "buffer_elements",
            "model_state_elements",
            "model_state_size_mib",
            "profiled_mflops_per_sample",
            "estimated_dominant_mmacs_per_sample",
            "recurrent_tokens_including_cls_dist",
            "gpu_latency_ms_median",
            "gpu_latency_ms_mean",
            "gpu_latency_ms_stdev",
            "gpu_throughput_samples_per_sec",
            "gpu_peak_memory_mib",
            "cpu_latency_ms_median",
            "cpu_latency_ms_mean",
            "cpu_latency_ms_stdev",
            "cpu_throughput_samples_per_sec",
        ]:
            summary[key] = first_row[key]

        summary_rows.append(summary)

    summary_fields = list(summary_rows[0].keys())
    summary_path = output_dir / "prc_seed_summary.csv"
    write_csv(summary_path, summary_rows, summary_fields)

    markdown_path = output_dir / "prc_seed_summary.md"
    with markdown_path.open("w", encoding="utf-8") as file:
        file.write("# PRC experiment summary\n\n")
        file.write(
            "| Method | Kept | Seeds | Accuracy (%) | Macro F1 (%) | "
            "FLOPs (M) | Footprint (MiB) | GPU latency (ms) | "
            "CPU latency (ms) |\n"
        )
        file.write(
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        )

        for row in summary_rows:
            accuracy = fmt(
                row["test_accuracy_percent_mean"],
                row["test_accuracy_percent_sd"],
            )
            f1 = fmt(
                row["test_f1_macro_percent_mean"],
                row["test_f1_macro_percent_sd"],
            )
            file.write(
                f"| {row['method']} | {row['keep_count']} | "
                f"{row['n_seeds']} | {accuracy} | {f1} | "
                f"{fmt_number(row['profiled_mflops_per_sample'])} | "
                f"{fmt_number(row['model_state_size_mib'])} | "
                f"{fmt_number(row['gpu_latency_ms_median'])} | "
                f"{fmt_number(row['cpu_latency_ms_median'])} |\n"
            )

    print(f"[OK] {detailed_path}")
    print(f"[OK] {summary_path}")
    print(f"[OK] {markdown_path}")


if __name__ == "__main__":
    main()
