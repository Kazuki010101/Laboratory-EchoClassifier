import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


TAG_PATTERN = re.compile(
    r"^(PRC_ce|PRC_kd|aps_ce|aps_kd|tg_skip|tgec)"
    r"(?:_ps(?P<patch_size>\d+)_k(?P<keep_count>\d+))?"
    r"_seed(?P<seed>\d+)(?P<smoke>_smoke)?$"
)

METHOD_NAMES = {
    "PRC_ce": "PRC-CE",
    "PRC_kd": "PRC-KD",
    "aps_ce": "APS-CE",
    "aps_kd": "APS-KD",
    "tg_skip": "TG-Skip",
    "tgec": "TGEC",
}

EFFICIENCY_KEYS = {
    "PRC_ce": "prc_full",
    "PRC_kd": "prc_full",
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


def upstream_summary_path(root, method, tag, seed, smoke):
    suffix = "_smoke" if smoke else ""
    if method == "PRC_ce":
        return root / "student_baseline" / f"PRC_ce_seed{seed}{suffix}" / "summary.json"
    if method == "PRC_kd":
        return root / "student_distillation" / "standard_kd" / f"PRC_seed{seed}{suffix}" / "summary.json"
    return root / "student_distillation" / "prc_condensation" / tag / "summary.json"


def load_efficiency(root, patch_size, keep_count):
    candidates = [
        root / "results" / f"prc_efficiency_ps{patch_size}_k{keep_count}.json",
    ]
    if patch_size == 16 and keep_count == 16:
        candidates += [
            root / "results" / "prc_efficiency_ps16_k16.json",
            root / "results" / "prc_efficiency.json",
        ]
    for path in candidates:
        if path.is_file():
            return read_json(path), path
    return {}, None


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_sd(values):
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None, None
    return statistics.mean(clean), statistics.stdev(clean) if len(clean) >= 2 else None


def fmt(mean, sd):
    if mean is None:
        return "-"
    if sd is None:
        return f"{mean:.3f}"
    return f"{mean:.3f} ± {sd:.3f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", default="experiments/shl2023_senvt_kd")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    root = Path(args.experiment_root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else root / "results"
    evaluation_root = root / "heldout_evaluation" / "user23_test"

    rows = []
    for evaluation_path in sorted(evaluation_root.glob("*/evaluation_summary.json")):
        tag = evaluation_path.parent.name
        match = TAG_PATTERN.match(tag)
        if not match:
            print(f"[WARN] Unrecognized result directory: {tag}")
            continue

        method = match.group(1)
        seed = int(match.group("seed"))
        smoke = bool(match.group("smoke"))
        patch_size = int(match.group("patch_size") or 16)
        keep_count = match.group("keep_count")
        if keep_count is None:
            keep_count = 31 if method.startswith("PRC_") else 16
        keep_count = int(keep_count)

        evaluation = read_json(evaluation_path)
        upstream_path = upstream_summary_path(root, method, tag, seed, smoke)
        finetune_path = root / "student_finetune" / "user23" / tag / "summary.json"
        upstream = read_json(upstream_path) if upstream_path.is_file() else {}
        finetune = read_json(finetune_path) if finetune_path.is_file() else {}

        efficiency_json, efficiency_path = load_efficiency(root, patch_size, keep_count)
        efficiency = efficiency_json.get(EFFICIENCY_KEYS[method], {})

        row = {
            "method": METHOD_NAMES[method],
            "tag": tag,
            "seed": seed,
            "patch_size": patch_size,
            "keep_count": first(efficiency, "keep_count") or keep_count,
            "keep_ratio": first(efficiency, "keep_ratio") or (keep_count / 31),
            "user1_val_accuracy": first(upstream, "best_acc1"),
            "user23_val_accuracy": first(finetune, "best_acc1"),
            "test_accuracy": first(evaluation, "acc1"),
            "test_precision_macro": first(evaluation, "precision_macro"),
            "test_recall_macro": first(evaluation, "recall_macro"),
            "test_f1_macro": first(evaluation, "f1_macro"),
            "test_loss": first(evaluation, "loss"),
            "test_samples": first(evaluation, "num_samples"),
            "route_entropy": first(evaluation, "routing_route_entropy", "route_entropy"),
            "trainable_parameters": first(efficiency, "trainable_parameters"),
            "model_state_elements": first(efficiency, "model_state_elements", "parameters"),
            "model_state_size_mb": first(efficiency, "model_state_size_mb"),
            "latency_ms_median": first(efficiency, "latency_ms_median"),
            "latency_ms_mean": first(efficiency, "latency_ms_mean"),
            "peak_memory_mb": first(efficiency, "peak_memory_mb"),
            "recurrent_tokens_including_cls_dist": first(
                efficiency, "recurrent_tokens_including_cls_dist"
            ),
            "estimated_dominant_macs": first(efficiency, "estimated_dominant_macs"),
            "efficiency_source": str(efficiency_path) if efficiency_path else "",
        }
        rows.append(row)

    if not rows:
        raise SystemExit(f"No evaluation_summary.json found under {evaluation_root}")

    detailed_fields = list(rows[0].keys())
    detailed_path = output_dir / "prc_combined_results.csv"
    write_csv(detailed_path, rows, detailed_fields)

    groups = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["patch_size"], int(float(row["keep_count"])))] .append(row)

    summary_rows = []
    for (method, patch_size, keep_count), group in sorted(groups.items()):
        summary = {
            "method": method,
            "patch_size": patch_size,
            "keep_count": keep_count,
            "n_seeds": len({row["seed"] for row in group}),
            "seeds": " ".join(str(x) for x in sorted({row["seed"] for row in group})),
        }
        for key in [
            "test_accuracy",
            "test_precision_macro",
            "test_recall_macro",
            "test_f1_macro",
        ]:
            mean, sd = mean_sd([row[key] for row in group])
            summary[f"{key}_mean"] = mean
            summary[f"{key}_sd"] = sd

        first_row = group[0]
        for key in [
            "trainable_parameters",
            "model_state_elements",
            "model_state_size_mb",
            "latency_ms_median",
            "latency_ms_mean",
            "peak_memory_mb",
            "recurrent_tokens_including_cls_dist",
            "estimated_dominant_macs",
        ]:
            summary[key] = first_row[key]
        summary_rows.append(summary)

    summary_fields = list(summary_rows[0].keys())
    summary_path = output_dir / "prc_seed_summary.csv"
    write_csv(summary_path, summary_rows, summary_fields)

    markdown_path = output_dir / "prc_seed_summary.md"
    with markdown_path.open("w", encoding="utf-8") as file:
        file.write("# PRC experiment summary\n\n")
        file.write("| Method | Patch size | Kept | Seeds | Accuracy (%) | Precision | Recall | Macro F1 | Latency median (ms) | MACs |\n")
        file.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in summary_rows:
            acc = fmt(row["test_accuracy_mean"], row["test_accuracy_sd"])
            precision = fmt(row["test_precision_macro_mean"], row["test_precision_macro_sd"])
            recall = fmt(row["test_recall_macro_mean"], row["test_recall_macro_sd"])
            f1 = fmt(row["test_f1_macro_mean"], row["test_f1_macro_sd"])
            latency = row["latency_ms_median"]
            latency_text = "-" if latency is None else f"{float(latency):.3f}"
            macs = row["estimated_dominant_macs"]
            macs_text = "-" if macs is None else str(int(float(macs)))
            file.write(
                f"| {row['method']} | {row['patch_size']} | {row['keep_count']} | "
                f"{row['n_seeds']} | {acc} | {precision} | {recall} | {f1} | "
                f"{latency_text} | {macs_text} |\n"
            )

    print(f"[OK] {detailed_path}")
    print(f"[OK] {summary_path}")
    print(f"[OK] {markdown_path}")


if __name__ == "__main__":
    main()

