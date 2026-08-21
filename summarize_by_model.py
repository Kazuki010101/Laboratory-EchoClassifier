import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    required_cols = [
        "method",
        "teacher",
        "student",
        "best_acc1",
        "best_epoch",
        "time_to_best",
        "elapsed_training_time",
        "max_memory_mb_all",
        "params",
        "params_total",
        "params_trainable",
        "size_mb",
        "flops_m",
        "latency_sample_ms",
        "throughput_samples_per_sec",
    ]

    keep_cols = [c for c in required_cols if c in df.columns]
    df = df[keep_cols].copy()

    def make_label(row):
        method = row.get("method", "")
        teacher = row.get("teacher", "")

        if method == "stage0_baseline":
            return "baseline"

        if method == "one_stage_direct":
            return "one_stage_B"

        if method == "stage2_two_stage" and teacher == "senvt-S":
            return "two_stage_S"

        if method == "stage2_two_stage" and teacher == "senvt-XS":
            return "two_stage_XS"

        if method == "reference_paper_style":
            return "reference_MLPMixer"

        if method == "stage1_intermediate_teacher":
            return "stage1_teacher"

        if method == "teacher_finetune":
            return "teacher_finetune"

        return f"{method}_{teacher}"

    df["setting"] = df.apply(make_label, axis=1)

    acc_table = df.pivot_table(
        index="student",
        columns="setting",
        values="best_acc1",
        aggfunc="max"
    ).reset_index()
    acc_table.columns.name = None

    epoch_table = df.pivot_table(
        index="student",
        columns="setting",
        values="best_epoch",
        aggfunc="max"
    ).reset_index()
    epoch_table.columns = [
        "student" if c == "student" else f"{c}_best_epoch"
        for c in epoch_table.columns
    ]

    memory_table = df.pivot_table(
        index="student",
        columns="setting",
        values="max_memory_mb_all",
        aggfunc="max"
    ).reset_index()
    memory_table.columns = [
        "student" if c == "student" else f"{c}_max_memory_mb"
        for c in memory_table.columns
    ]

    time_table = df.pivot_table(
        index="student",
        columns="setting",
        values="elapsed_training_time",
        aggfunc="max"
    ).reset_index()
    time_table.columns = [
        "student" if c == "student" else f"{c}_elapsed_time"
        for c in time_table.columns
    ]

    fp_cols = [
        "student",
        "params",
        "params_total",
        "params_trainable",
        "size_mb",
        "flops_m",
        "latency_sample_ms",
        "throughput_samples_per_sec",
    ]
    fp_cols = [c for c in fp_cols if c in df.columns]

    footprint = (
        df[fp_cols]
        .dropna(subset=["student"])
        .drop_duplicates(subset=["student"])
    )

    out = acc_table.merge(epoch_table, on="student", how="left")
    out = out.merge(memory_table, on="student", how="left")
    out = out.merge(time_table, on="student", how="left")
    out = out.merge(footprint, on="student", how="left")

    preferred_order = [
        "student",

        "baseline",
        "one_stage_B",
        "two_stage_S",
        "two_stage_XS",
        "reference_MLPMixer",

        "baseline_max_memory_mb",
        "one_stage_B_max_memory_mb",
        "two_stage_S_max_memory_mb",
        "two_stage_XS_max_memory_mb",

        "params",
        "params_total",
        "params_trainable",
        "size_mb",
        "flops_m",
        "latency_sample_ms",
        "throughput_samples_per_sec",
    ]

    ordered_cols = [c for c in preferred_order if c in out.columns]
    other_cols = [c for c in out.columns if c not in ordered_cols]

    out = out[ordered_cols + other_cols]
    out = out.sort_values("student")

    out.to_csv(args.output_csv, index=False)

    print(f"[done] wrote: {args.output_csv}")
    print(f"[done] rows: {len(out)}")


if __name__ == "__main__":
    main()