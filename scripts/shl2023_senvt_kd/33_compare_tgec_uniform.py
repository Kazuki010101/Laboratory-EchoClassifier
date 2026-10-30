
import json
import csv
import statistics
from pathlib import Path

ROOT = Path(
    "experiments/shl2023_senvt_kd/"
    "heldout_evaluation/user23_test"
)

OUTPUT = Path(
    "experiments/shl2023_senvt_kd/"
    "results/tgec_uniform_comparison.csv"
)

METHODS = {
    "tgec": "TGEC (ours)",
    "tgec_uniform": "TGEC-Uniform",
}

METRICS = {
    "acc1": "Accuracy",
    "precision_macro": "Macro Precision",
    "recall_macro": "Macro Recall",
    "f1_macro": "Macro F1",
}

rows = []

for method, label in METHODS.items():
    values = {name: [] for name in METRICS.values()}
    for seed in (0, 1, 2):
        path = ROOT / f"{method}_seed{seed}" / "evaluation_summary.json"

        if not path.exists():
            raise FileNotFoundError(f"Missing result: {path}")

        with path.open(encoding="utf-8") as f:
            result = json.load(f)

        row = {"Method": label, "Seed": seed}

        for key, name in METRICS.items():
            if key not in result:
                raise KeyError(
                    f"{key} not found in {path}. "
                    f"Available keys: {list(result.keys())}"
                )

            v = float(result[key])
            if abs(v) <= 1:
                v *= 100

            values[name].append(v)
            row[name] = round(v, 4)

        rows.append(row)

    print(f"\n=== {label} ===")

    for name, v in values.items():
        mean = statistics.mean(v)
        sd = statistics.stdev(v)
        print(f"{name:18s}: {mean:.4f} ± {sd:.4f}%")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "Method", "Seed",
            "Accuracy",
            "Macro Precision",
            "Macro Recall",
            "Macro F1",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSaved: {OUTPUT}")
