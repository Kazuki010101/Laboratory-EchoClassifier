#!/usr/bin/env python3
import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


TAG = re.compile(r"^(prc_ce|prc_kd|aps_ce|aps_kd|tg_skip|tgec)_seed(\d+)$")
DISPLAY = {
    "prc_ce": "PRC-CE", "prc_kd": "PRC-KD", "aps_ce": "APS-CE",
    "aps_kd": "APS-KD", "tg_skip": "TG-Skip", "tgec": "TGEC",
}


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(targets, predictions):
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predictions, labels=np.arange(12), average="macro", zero_division=0
    )
    return {
        "accuracy": 100.0 * accuracy_score(targets, predictions),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }


def mean_sd(values):
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-folds", nargs="+", type=int, default=list(range(101, 109)))
    parser.add_argument("--expected-seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()

    root = Path(args.evaluation_root).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    fold_rows = []
    pooled = defaultdict(lambda: {"targets": [], "predictions": [], "folds": []})

    for summary_path in sorted(root.glob("test*/*/evaluation_summary.json")):
        try:
            fold = int(summary_path.parents[1].name.removeprefix("test"))
        except ValueError:
            continue
        match = TAG.match(summary_path.parent.name)
        if not match:
            continue
        method, seed_text = match.groups()
        seed = int(seed_text)
        pred_path = summary_path.parent / "predictions.npz"
        if not pred_path.is_file():
            raise FileNotFoundError(f"Missing predictions: {pred_path}")
        values = np.load(pred_path)
        predictions = np.asarray(values["predictions"])
        targets = np.asarray(values["targets"])
        if len(predictions) != len(targets):
            raise ValueError(f"Prediction count mismatch: {pred_path}")
        score = metrics(targets, predictions)
        fold_rows.append({
            "method": DISPLAY[method], "method_key": method, "seed": seed,
            "test_subject": fold, "num_samples": len(targets), **score,
        })
        group = pooled[(method, seed)]
        group["targets"].append(targets)
        group["predictions"].append(predictions)
        group["folds"].append(fold)

    if not fold_rows:
        raise SystemExit(f"No complete evaluation results found under {root}")

    seed_rows = []
    for (method, seed), group in sorted(pooled.items()):
        folds = sorted(group["folds"])
        missing = sorted(set(args.expected_folds) - set(folds))
        score = metrics(np.concatenate(group["targets"]), np.concatenate(group["predictions"]))
        seed_rows.append({
            "method": DISPLAY[method], "method_key": method, "seed": seed,
            "n_folds": len(folds), "folds": " ".join(map(str, folds)),
            "complete": not missing, "missing_folds": " ".join(map(str, missing)),
            "num_samples": sum(map(len, group["targets"])), **score,
        })

    by_method = defaultdict(list)
    for row in seed_rows:
        if row["complete"] and row["seed"] in args.expected_seeds:
            by_method[row["method_key"]].append(row)

    summary_rows = []
    for method in DISPLAY:
        rows = sorted(by_method.get(method, []), key=lambda row: row["seed"])
        if not rows:
            continue
        summary = {
            "method": DISPLAY[method],
            "n_seeds": len(rows),
            "seeds": " ".join(str(row["seed"]) for row in rows),
        }
        for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
            mean, sd = mean_sd([float(row[key]) for row in rows])
            summary[f"{key}_mean"] = mean
            summary[f"{key}_sd"] = sd
        summary_rows.append(summary)

    write_csv(output / "pamap2_fold_results.csv", fold_rows)
    write_csv(output / "pamap2_pooled_seed_results.csv", seed_rows)
    write_csv(output / "pamap2_seed_summary.csv", summary_rows)
    (output / "pamap2_collection_status.json").write_text(
        json.dumps({"expected_folds": args.expected_folds, "expected_seeds": args.expected_seeds,
                    "pooled_runs": seed_rows}, indent=2), encoding="utf-8"
    )

    with (output / "pamap2_seed_summary.md").open("w", encoding="utf-8") as file:
        file.write("# PAMAP2 pooled LOSO results\n\n")
        file.write("| Method | Seeds | Accuracy (%) | Macro Precision | Macro Recall | Macro F1 |\n")
        file.write("|---|---:|---:|---:|---:|---:|\n")
        for row in summary_rows:
            def fmt(key):
                mean, sd = row[f"{key}_mean"], row[f"{key}_sd"]
                return f"{mean:.3f}" if sd is None else f"{mean:.3f} ± {sd:.3f}"
            file.write(
                f"| {row['method']} | {row['n_seeds']} | {fmt('accuracy')} | "
                f"{fmt('precision_macro')} | {fmt('recall_macro')} | {fmt('f1_macro')} |\n"
            )
    for name in ("pamap2_fold_results.csv", "pamap2_pooled_seed_results.csv",
                 "pamap2_seed_summary.csv", "pamap2_seed_summary.md",
                 "pamap2_collection_status.json"):
        print(f"[OK] {output / name}")


if __name__ == "__main__":
    main()

