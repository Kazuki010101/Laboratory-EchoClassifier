import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None

    try:
        with path.open("r") as f:
            return json.load(f)
    except Exception:
        return None


def read_log_json_lines(log_path: Path) -> List[dict]:
    records = []

    if not log_path.exists():
        return records

    with log_path.open("r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                records.append(json.loads(line))
            except Exception:
                continue

    return records


def best_from_log(log_path: Path) -> Dict[str, Optional[float]]:
    records = read_log_json_lines(log_path)

    if not records:
        return {
            "best_acc1": None,
            "best_epoch": None,
            "last_acc1": None,
            "last_epoch": None,
            "time_to_best": None,
            "elapsed_training_time": None,
            "max_memory_mb_all": None,
        }

    best_record = None
    best_acc = -1.0

    for r in records:
        acc = r.get("test_acc1", None)

        if acc is not None and acc > best_acc:
            best_acc = acc
            best_record = r

    last = records[-1]

    return {
        "best_acc1": best_acc if best_record is not None else None,
        "best_epoch": best_record.get("epoch") if best_record is not None else None,
        "last_acc1": last.get("test_acc1"),
        "last_epoch": last.get("epoch"),
        "time_to_best": best_record.get("time_to_best") if best_record is not None else None,
        "elapsed_training_time": last.get("elapsed_training_time"),
        "max_memory_mb_all": last.get("max_memory_mb_all"),
    }


def infer_group_and_model(exp_root: Path, run_dir: Path) -> Tuple[str, str, str, str]:
    rel = run_dir.relative_to(exp_root)
    parts = rel.parts

    method = parts[0] if len(parts) > 0 else ""
    teacher = ""
    student_name = run_dir.name

    if method == "stage0_baseline":
        teacher = "none"
        student_name = run_dir.name.replace("no_kd_to_", "")

    elif method == "teacher_finetune":
        teacher = "none"
        student_name = run_dir.name

    elif method == "one_stage_direct":
        teacher = "senvt-B"
        student_name = run_dir.name.replace("senvtB_to_", "")

    elif method == "stage1_intermediate_teacher":
        teacher = "senvt-B"

        if "XS" in run_dir.name:
            student_name = "senvt-XS"
        elif "S" in run_dir.name:
            student_name = "senvt-S"

    elif method == "stage2_two_stage":
        if len(parts) >= 2:
            parent = parts[1]

            if "senvtS" in parent:
                teacher = "senvt-S"
                student_name = run_dir.name.replace("senvtS_to_", "")
            elif "senvtXS" in parent:
                teacher = "senvt-XS"
                student_name = run_dir.name.replace("senvtXS_to_", "")

    elif method == "reference_paper_style":
        teacher = "MLPMixer"
        student_name = run_dir.name.replace("mlpmixer_to_", "")

    return method, teacher, student_name, str(rel)


def load_footprint_csv(path: Path) -> Dict[str, dict]:
    footprints = {}

    if not path.exists():
        print(f"[warn] footprint csv not found: {path}")
        return footprints

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            model_key = row.get("model_key", "")
            model = row.get("model", "")
            patch_size = row.get("patch_size", "")
            reservoir_size = row.get("reservoir_size", "")
            reservoir_rank = row.get("reservoir_rank", "")

            keys = []

            if model_key:
                keys.append(model_key)

            if model:
                keys.append(model)

            if model in ["PRC", "PESAC"] and patch_size and reservoir_size:
                keys.append(f"{model}_p{patch_size}_r{reservoir_size}")

            if model == "PRC_LRGR" and patch_size and reservoir_size and reservoir_rank:
                keys.append(f"LRGR_p{patch_size}_r{reservoir_size}_rank{reservoir_rank}")

            for key in keys:
                if key:
                    footprints[key] = row

    return footprints


def find_run_dirs(exp_root: Path) -> List[Path]:
    run_dirs = []

    target_files = ["summary.json", "log.txt"]

    for root, dirs, files in os.walk(exp_root):
        root_path = Path(root)

        if "footprint" in root_path.parts:
            continue

        if any(t in files for t in target_files):
            run_dirs.append(root_path)

    return sorted(run_dirs)


def safe_get(d: Optional[dict], key: str, default=""):
    if d is None:
        return default

    value = d.get(key, default)

    return "" if value is None else value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-root", required=True)
    parser.add_argument("--footprint-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    exp_root = Path(args.exp_root)
    footprint_csv = Path(args.footprint_csv)
    output_csv = Path(args.output_csv)

    footprints = load_footprint_csv(footprint_csv)
    run_dirs = find_run_dirs(exp_root)

    rows = []

    for run_dir in run_dirs:
        method, teacher, student_name, rel_path = infer_group_and_model(
            exp_root=exp_root,
            run_dir=run_dir
        )

        summary = load_json(run_dir / "summary.json")
        log_stats = best_from_log(run_dir / "log.txt")

        best_acc1 = safe_get(summary, "best_acc1", "")
        if best_acc1 == "":
            best_acc1 = log_stats.get("best_acc1", "")

        best_epoch = safe_get(summary, "best_epoch", "")
        if best_epoch == "":
            best_epoch = log_stats.get("best_epoch", "")

        time_to_best = safe_get(summary, "time_to_best", "")
        if time_to_best == "":
            time_to_best = log_stats.get("time_to_best", "")

        elapsed_training_time = safe_get(summary, "elapsed_training_time", "")
        if elapsed_training_time == "":
            elapsed_training_time = log_stats.get("elapsed_training_time", "")

        max_memory_mb_all = safe_get(summary, "max_memory_mb_all", "")
        if max_memory_mb_all == "":
            max_memory_mb_all = log_stats.get("max_memory_mb_all", "")

        fp = footprints.get(student_name, {})

        row = {
            "method": method,
            "teacher": teacher,
            "student": student_name,
            "run_path": rel_path,

            "best_acc1": best_acc1,
            "best_epoch": best_epoch,
            "time_to_best": time_to_best,
            "elapsed_training_time": elapsed_training_time,
            "max_memory_mb_all": max_memory_mb_all,

            "params": fp.get("params", ""),
            "params_total": fp.get("params_total", ""),
            "params_trainable": fp.get("params_trainable", ""),
            "size_mb": fp.get("size_mb", ""),
            "flops": fp.get("flops", ""),
            "flops_m": fp.get("flops_m", ""),
            "latency_batch_ms": fp.get("latency_batch_ms", ""),
            "latency_sample_ms": fp.get("latency_sample_ms", ""),
            "throughput_samples_per_sec": fp.get("throughput_samples_per_sec", ""),

            "footprint_model_key": fp.get("model_key", ""),
            "footprint_model": fp.get("model", ""),
            "footprint_patch_size": fp.get("patch_size", ""),
            "footprint_reservoir_size": fp.get("reservoir_size", ""),
            "footprint_reservoir_rank": fp.get("reservoir_rank", ""),
        }

        rows.append(row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "method",
        "teacher",
        "student",
        "run_path",

        "best_acc1",
        "best_epoch",
        "time_to_best",
        "elapsed_training_time",
        "max_memory_mb_all",

        "params",
        "params_total",
        "params_trainable",
        "size_mb",
        "flops",
        "flops_m",
        "latency_batch_ms",
        "latency_sample_ms",
        "throughput_samples_per_sec",

        "footprint_model_key",
        "footprint_model",
        "footprint_patch_size",
        "footprint_reservoir_size",
        "footprint_reservoir_rank",
    ]

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    missing_fp = [
        r["student"]
        for r in rows
        if r.get("params", "") == "" and r.get("size_mb", "") == ""
    ]

    print(f"[done] wrote: {output_csv}")
    print(f"[done] rows: {len(rows)}")
    print(f"[done] footprint rows loaded: {len(footprints)}")

    if missing_fp:
        print("[warn] these students have no footprint matched:")
        for name in sorted(set(missing_fp)):
            print(f"  - {name}")


if __name__ == "__main__":
    main()