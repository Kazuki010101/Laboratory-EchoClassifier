import argparse
import csv
import json
from pathlib import Path
# /*
# python transfer_scripts/summarize_pamap2_transfer.py \
#   --root experiments/SHL2023_senvt_echo_w496/pamap2_transfer \
#   --output experiments/SHL2023_senvt_echo_w496/pamap2_transfer_summary.csv
# */
def load_json(path: Path):
    try:
        with path.open("r") as f:
            return json.load(f)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="experiments/SHL2023_senvt_echo_w496/pamap2_transfer",
    )
    parser.add_argument(
        "--output",
        default="experiments/SHL2023_senvt_echo_w496/pamap2_transfer_summary.csv",
    )
    args = parser.parse_args()

    root = Path(args.root)
    rows = []

    for test_json in sorted(root.glob("*/*/test_best.json")):
        data = load_json(test_json)
        if data is None:
            continue

        run_dir = test_json.parent
        model_name = run_dir.parent.name

        row = {
            "model_name": model_name,
            "run_path": str(run_dir),
            "student": data.get("student", ""),
            "patch_size": data.get("patch_size", ""),
            "reservoir_size": data.get("reservoir_size", ""),
            "reservoir_rank": data.get("reservoir_rank", ""),
            "patch_keep_ratio": data.get("patch_keep_ratio", ""),
            "test_subject": data.get("test_subject", ""),
            "val_subject": data.get("val_subject", ""),
            "best_epoch": data.get("best_epoch", ""),
            "best_val_loss": data.get("best_val_loss", ""),
            "best_val_acc1": data.get("best_val_acc1", ""),
            "test_loss": data.get("test_loss", ""),
            "test_acc1": data.get("test_acc1", ""),
            "test_f1_macro": data.get("test_f1_macro", ""),
            "test_precision_macro": data.get("test_precision_macro", ""),
            "test_recall_macro": data.get("test_recall_macro", ""),
            "loaded_transfer_keys": data.get("loaded_transfer_keys", ""),
            "skipped_transfer_keys": data.get("skipped_transfer_keys", ""),
            "transfer_checkpoint": data.get("transfer_checkpoint", ""),
        }

        rows.append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "model_name",
        "run_path",
        "student",
        "patch_size",
        "reservoir_size",
        "reservoir_rank",
        "patch_keep_ratio",
        "test_subject",
        "val_subject",
        "best_epoch",
        "best_val_loss",
        "best_val_acc1",
        "test_loss",
        "test_acc1",
        "test_f1_macro",
        "test_precision_macro",
        "test_recall_macro",
        "loaded_transfer_keys",
        "skipped_transfer_keys",
        "transfer_checkpoint",
    ]

    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[done] wrote: {output}")
    print(f"[done] rows: {len(rows)}")


if __name__ == "__main__":
    main()