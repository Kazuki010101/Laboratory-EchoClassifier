import argparse
import csv
import json
import statistics
from pathlib import Path


DETAIL_FIELDS = [
    "model_name",
    "test_subject",
    "baseline_acc",
    "transfer_acc",
    "delta_acc",
]

SUMMARY_FIELDS = [
    "model_name",
    "baseline_acc_mean",
    "baseline_acc_std",
    "baseline_worst_subject",
    "baseline_worst_acc",
    "transfer_acc_mean",
    "transfer_acc_std",
    "transfer_worst_subject",
    "transfer_worst_acc",
    "delta_acc_mean",
    "delta_acc_std",
    "delta_worst_subject",
    "delta_worst_acc",
    "params_total",
    "params_trainable",
    "size_mb",
    "flops_m",
    "latency_sample_ms",
    "throughput_samples_per_sec",
]


def load_json(path: Path):
    try:
        with path.open("r") as f:
            return json.load(f)
    except Exception:
        return None


def read_csv_rows(path: Path):
    if not path.exists():
        return []

    try:
        with path.open("r", newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def to_float(value):
    if value is None or value == "":
        return None

    try:
        return float(value)
    except Exception:
        return None


def rounded(value, digits=4):
    if value is None or value == "":
        return ""

    try:
        return round(float(value), digits)
    except Exception:
        return value


def mean(values):
    nums = [to_float(v) for v in values]
    nums = [v for v in nums if v is not None]

    if not nums:
        return ""

    return sum(nums) / len(nums)


def std(values):
    nums = [to_float(v) for v in values]
    nums = [v for v in nums if v is not None]

    if len(nums) <= 1:
        return ""

    return statistics.stdev(nums)


def diff(transfer_value, baseline_value):
    transfer = to_float(transfer_value)
    baseline = to_float(baseline_value)

    if transfer is None or baseline is None:
        return ""

    return rounded(transfer - baseline)


def pick_value(row, candidates):
    for key in candidates:
        if key in row and row[key] not in [None, ""]:
            return row[key]

    return ""


def normalize_model_name(name):
    return str(name).strip()


def load_footprint_map(footprint_csv: Path):
    rows = read_csv_rows(footprint_csv)
    footprint = {}

    for row in rows:
        model_name = pick_value(
            row,
            ["model_key", "run_name", "name", "model_name", "student"],
        )
        model_name = normalize_model_name(model_name)

        if not model_name:
            continue

        footprint[model_name] = {
            "params_total": pick_value(
                row,
                ["params_total", "total_params", "n_parameters_total", "params"],
            ),
            "params_trainable": pick_value(
                row,
                ["params_trainable", "trainable_params", "n_parameters"],
            ),
            "size_mb": pick_value(
                row,
                ["size_mb", "model_size_mb", "footprint_size_mb"],
            ),
            "flops_m": pick_value(
                row,
                ["flops_m", "FLOPs_M", "flops"],
            ),
            "latency_sample_ms": pick_value(
                row,
                ["latency_sample_ms", "latency_ms", "avg_latency_ms"],
            ),
            "throughput_samples_per_sec": pick_value(
                row,
                ["throughput_samples_per_sec", "throughput", "samples_per_sec"],
            ),
        }

    return footprint


def find_test_json_files(root: Path):
    files = sorted(root.glob("*/*/test_best.json"))

    if files:
        return files

    return sorted(root.glob("**/test_best.json"))


def read_result_rows(root: Path, condition: str, exclude_subjects):
    rows = []

    for test_json in find_test_json_files(root):
        data = load_json(test_json)

        if data is None:
            continue

        run_dir = test_json.parent
        model_name = normalize_model_name(run_dir.parent.name)
        test_subject = str(data.get("test_subject", ""))

        if test_subject in exclude_subjects:
            continue

        row = {
            "condition": condition,
            "model_name": model_name,
            "test_subject": test_subject,
            "acc": data.get("test_acc1", ""),
        }

        rows.append(row)

    return rows


def make_pair_key(row):
    return (
        normalize_model_name(row.get("model_name", "")),
        str(row.get("test_subject", "")),
    )


def make_detail_rows(baseline_rows, transfer_rows):
    baseline_map = {make_pair_key(row): row for row in baseline_rows}
    transfer_map = {make_pair_key(row): row for row in transfer_rows}

    all_keys = sorted(
        set(baseline_map.keys()) | set(transfer_map.keys()),
        key=lambda x: (x[0], int(x[1]) if str(x[1]).isdigit() else str(x[1])),
    )

    detail_rows = []

    for model_name, test_subject in all_keys:
        baseline = baseline_map.get((model_name, test_subject))
        transfer = transfer_map.get((model_name, test_subject))

        baseline_acc = baseline.get("acc", "") if baseline else ""
        transfer_acc = transfer.get("acc", "") if transfer else ""

        row = {
            "model_name": model_name,
            "test_subject": test_subject,
            "baseline_acc": baseline_acc,
            "transfer_acc": transfer_acc,
            "delta_acc": diff(transfer_acc, baseline_acc),
        }

        detail_rows.append(row)

    return detail_rows


def group_by_model(rows):
    groups = {}

    for row in rows:
        model_name = normalize_model_name(row.get("model_name", ""))
        groups.setdefault(model_name, []).append(row)

    return groups


def find_worst_subject(rows, acc_field):
    candidates = []

    for row in rows:
        acc = to_float(row.get(acc_field, ""))
        subject = str(row.get("test_subject", ""))

        if acc is None or subject == "":
            continue

        candidates.append((acc, subject))

    if not candidates:
        return "", ""

    worst_acc, worst_subject = min(candidates, key=lambda x: x[0])

    return worst_subject, rounded(worst_acc)


def make_summary_rows(detail_rows, footprint_map):
    groups = group_by_model(detail_rows)
    summary_rows = []

    for model_name, rows in sorted(groups.items()):
        baseline_rows = [row for row in rows if to_float(row.get("baseline_acc", "")) is not None]
        transfer_rows = [row for row in rows if to_float(row.get("transfer_acc", "")) is not None]
        paired_rows = [
            row for row in rows
            if to_float(row.get("baseline_acc", "")) is not None
            and to_float(row.get("transfer_acc", "")) is not None
        ]

        baseline_worst_subject, baseline_worst_acc = find_worst_subject(
            baseline_rows,
            "baseline_acc",
        )
        transfer_worst_subject, transfer_worst_acc = find_worst_subject(
            transfer_rows,
            "transfer_acc",
        )
        delta_worst_subject, delta_worst_acc = find_worst_subject(
            paired_rows,
            "delta_acc",
        )

        energy = footprint_map.get(model_name, {})

        row = {
            "model_name": model_name,
            "baseline_acc_mean": rounded(mean([r.get("baseline_acc") for r in baseline_rows])),
            "baseline_acc_std": rounded(std([r.get("baseline_acc") for r in baseline_rows])),
            "baseline_worst_subject": baseline_worst_subject,
            "baseline_worst_acc": baseline_worst_acc,
            "transfer_acc_mean": rounded(mean([r.get("transfer_acc") for r in transfer_rows])),
            "transfer_acc_std": rounded(std([r.get("transfer_acc") for r in transfer_rows])),
            "transfer_worst_subject": transfer_worst_subject,
            "transfer_worst_acc": transfer_worst_acc,
            "delta_acc_mean": rounded(mean([r.get("delta_acc") for r in paired_rows])),
            "delta_acc_std": rounded(std([r.get("delta_acc") for r in paired_rows])),
            "delta_worst_subject": delta_worst_subject,
            "delta_worst_acc": delta_worst_acc,
            "params_total": energy.get("params_total", ""),
            "params_trainable": energy.get("params_trainable", ""),
            "size_mb": energy.get("size_mb", ""),
            "flops_m": energy.get("flops_m", ""),
            "latency_sample_ms": energy.get("latency_sample_ms", ""),
            "throughput_samples_per_sec": energy.get("throughput_samples_per_sec", ""),
        }

        summary_rows.append(row)

    return summary_rows


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--baseline-root",
        default="experiments/SHL2023_senvt_echo_w496/pamap2_baseline_all_models",
    )
    parser.add_argument(
        "--transfer-root",
        default="experiments/SHL2023_senvt_echo_w496/pamap2_transfer",
    )
    parser.add_argument(
        "--footprint-csv",
        default="experiments/SHL2023_senvt_echo_w496/footprint/model_static_metrics.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/SHL2023_senvt_echo_w496/pamap2_compare_all_models",
    )
    parser.add_argument(
        "--exclude-subjects",
        nargs="*",
        default=["102"],
    )

    args = parser.parse_args()

    baseline_root = Path(args.baseline_root)
    transfer_root = Path(args.transfer_root)
    footprint_csv = Path(args.footprint_csv)
    output_dir = Path(args.output_dir)
    exclude_subjects = set(str(s) for s in args.exclude_subjects)

    footprint_map = load_footprint_map(footprint_csv)

    baseline_rows = read_result_rows(
        root=baseline_root,
        condition="pamap2_baseline",
        exclude_subjects=exclude_subjects,
    )
    transfer_rows = read_result_rows(
        root=transfer_root,
        condition="shl_to_pamap2_transfer",
        exclude_subjects=exclude_subjects,
    )

    detail_rows = make_detail_rows(
        baseline_rows=baseline_rows,
        transfer_rows=transfer_rows,
    )
    summary_rows = make_summary_rows(
        detail_rows=detail_rows,
        footprint_map=footprint_map,
    )

    detail_path = output_dir / "pamap2_compact_subject_detail.csv"
    summary_path = output_dir / "pamap2_compact_model_summary.csv"

    write_csv(detail_path, detail_rows, DETAIL_FIELDS)
    write_csv(summary_path, summary_rows, SUMMARY_FIELDS)

    print("===== PAMAP2 Compact Summary Done =====")
    print(f"excluded subjects: {' '.join(sorted(exclude_subjects))}")
    print(f"baseline rows: {len(baseline_rows)}")
    print(f"transfer rows: {len(transfer_rows)}")
    print(f"models: {len(summary_rows)}")
    print(f"footprint rows: {len(footprint_map)}")
    print(f"detail: {detail_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()