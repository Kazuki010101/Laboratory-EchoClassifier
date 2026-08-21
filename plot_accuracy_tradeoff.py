import os
import re
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =========================================================
# 1. Path
# =========================================================
summary = "experiments/SHL2023_senvt_echo_w496/pamap2_compare_all_models/pamap2_compact_model_summary.csv"

out_dir = "experiments/SHL2023_senvt_echo_w496/pamap2_compare_all_models/figures_tradeoff_slide"
os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv(summary)

# =========================================================
# 2. Exclude models
# =========================================================
# IFSA / LRGR / PESAC 系を除外
df = df[~df["model_name"].str.match(r"^(IFSA_LRGR|LRGR|PESAC)")].copy()

# r4000 系を除外
df = df[~df["model_name"].str.contains("_r4000", regex=False)].copy()

# =========================================================
# 3. Accuracy column
# =========================================================
ACC_COL = "transfer_acc_mean"
# ACC_COL = "baseline_acc_mean"

# =========================================================
# 4. Slide style
# =========================================================
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 23,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
    "legend.title_fontsize": 15,
    "axes.linewidth": 1.4,
})

# =========================================================
# 5. Model family grouping
# =========================================================
def get_family(model_name: str) -> str:
    if model_name.startswith("APS_LRGR"):
        return "APS-LRGR"
    elif model_name.startswith("PRC"):
        return "PRC"
    elif model_name.startswith("DeepConvLSTM"):
        return "DeepConvLSTM"
    elif model_name.startswith("Resnet") or model_name.startswith("ResNet"):
        return "ResNet"
    elif model_name.startswith("MLPMixer"):
        return "MLP-Mixer"
    else:
        return "Others"

df["family"] = df["model_name"].apply(get_family)

# =========================================================
# 6. Colors and markers
# =========================================================
family_colors = {
    "APS-LRGR": "red",
    "PRC": "tab:green",
    "DeepConvLSTM": "tab:brown",
    "ResNet": "tab:cyan",
    "MLP-Mixer": "black",
    "Others": "gray",
}

family_markers = {
    "APS-LRGR": "o",
    "PRC": "s",
    "DeepConvLSTM": "^",
    "ResNet": "D",
    "MLP-Mixer": "X",
    "Others": "o",
}

# =========================================================
# 7. Full-ish label
#    省略しすぎず、プレゼンで読める形にする
# =========================================================
def short_label(name: str) -> str:
    # APS-LRGR
    if name.startswith("APS_LRGR_"):
        m = re.search(r"p(\d+)_r(\d+)_rank(\d+)_keep(\d+)", name)
        if m:
            p, r, rank, keep = m.groups()
            return f"APS-LRGR p{p}"
        return name.replace("APS_LRGR_", "APS-LRGR ")

    # PRC
    if name.startswith("PRC_"):
        m = re.search(r"p(\d+)_r(\d+)", name)
        if m:
            p, r = m.groups()
            return f"PRC p{p}"
        return name

    # DeepConvLSTM
    if name.startswith("DeepConvLSTM"):
        m = re.search(r"DeepConvLSTM(\d+)", name)
        if m:
            return f"DeepConvLSTM-{m.group(1)}"
        return name

    # ResNet
    if name.startswith("Resnet_"):
        return name.replace("Resnet_", "ResNet-")

    if name.startswith("ResNet_"):
        return name.replace("ResNet_", "ResNet-")

    # MLP-Mixer
    if name == "MLPMixer":
        return "MLPMixer"

    return name

# =========================================================
# 8. Label placement
# =========================================================
def add_non_overlapping_label(
    fig,
    ax,
    x,
    y,
    text,
    used_bboxes,
    family,
):
    candidate_offsets = [
        (10, 8),
        (10, -12),
        (-95, 8),
        (-95, -12),
        (12, 22),
        (-110, 22),
        (12, -26),
        (-110, -26),
        (28, 0),
        (-125, 0),
        (0, 36),
        (0, -40),
    ]

    if family == "APS-LRGR":
        fontsize = 14
        fontweight = "bold"
    elif family in ["PRC", "DeepConvLSTM", "ResNet", "MLP-Mixer"]:
        fontsize = 12
        fontweight = "bold"
    else:
        fontsize = 11
        fontweight = "normal"

    renderer = fig.canvas.get_renderer()

    for dx, dy in candidate_offsets:
        ann = ax.annotate(
            text,
            (x, y),
            textcoords="offset points",
            xytext=(dx, dy),
            ha="left",
            va="center",
            fontsize=fontsize,
            fontweight=fontweight,
            clip_on=False,
            bbox=dict(
                boxstyle="round,pad=0.22",
                fc="white",
                ec="none",
                alpha=0.88,
            ),
            arrowprops=dict(
                arrowstyle="-",
                lw=0.55,
                color="gray",
                alpha=0.65,
                shrinkA=2,
                shrinkB=4,
            ),
            zorder=20,
        )

        fig.canvas.draw()
        bbox = ann.get_window_extent(renderer=renderer).expanded(1.03, 1.15)
        overlap = any(bbox.overlaps(prev) for prev in used_bboxes)

        if not overlap:
            used_bboxes.append(bbox)
            return ann

        ann.remove()

    # 最後の逃げ
    ann = ax.annotate(
        text,
        (x, y),
        textcoords="offset points",
        xytext=(14, 42),
        ha="left",
        va="center",
        fontsize=fontsize,
        fontweight=fontweight,
        clip_on=False,
        bbox=dict(
            boxstyle="round,pad=0.22",
            fc="white",
            ec="none",
            alpha=0.88,
        ),
        arrowprops=dict(
            arrowstyle="-",
            lw=0.55,
            color="gray",
            alpha=0.65,
            shrinkA=2,
            shrinkB=4,
        ),
        zorder=20,
    )

    fig.canvas.draw()
    used_bboxes.append(ann.get_window_extent(renderer=renderer).expanded(1.03, 1.15))
    return ann

# =========================================================
# 9. Common plot function
# =========================================================
def plot_tradeoff_slide(
    df,
    x_col,
    x_label,
    filename,
    log_x=True,
    label_mode="all",
):
    """
    label_mode:
        "all"      : 全モデルにラベル
        "aps_only" : APS-LRGRだけラベル
        "none"     : ラベルなし
    """

    plot_df = df.dropna(subset=[x_col, ACC_COL]).copy()
    plot_df = plot_df[plot_df[x_col] > 0].copy()

    fig, ax = plt.subplots(figsize=(14, 8))

    for family, g in plot_df.groupby("family"):
        g = g.sort_values(by=x_col)

        if family == "APS-LRGR":
            size = 240
            alpha = 1.0
            linewidth = 1.5
            zorder = 6
        elif family == "PRC":
            size = 210
            alpha = 0.92
            linewidth = 1.3
            zorder = 5
        else:
            size = 185
            alpha = 0.82
            linewidth = 1.1
            zorder = 3

        ax.scatter(
            g[x_col],
            g[ACC_COL],
            s=size,
            alpha=alpha,
            color=family_colors.get(family, "gray"),
            marker=family_markers.get(family, "o"),
            edgecolor="black",
            linewidth=linewidth,
            label=family,
            zorder=zorder,
        )

    if log_x:
        ax.set_xscale("log")

    ax.set_xlabel(x_label, fontsize=20, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=20, fontweight="bold")

    title_acc = "Transfer Accuracy" if ACC_COL == "transfer_acc_mean" else "Baseline Accuracy"
    ax.set_title(f"{title_acc} vs {x_label}", fontsize=23, fontweight="bold", pad=16)

    ax.grid(True, which="major", linestyle="--", alpha=0.32, zorder=0)
    ax.minorticks_off()
    ax.tick_params(axis="both", which="major", labelsize=16, width=1.4, length=6)

    # 余白
    ax.margins(x=0.30, y=0.26)

    y_min = plot_df[ACC_COL].min()
    y_max = plot_df[ACC_COL].max()
    y_margin = max((y_max - y_min) * 0.26, 2.0)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)

    fig.canvas.draw()
    used_bboxes = []

    # APS優先で、その後に他モデル
    label_df = plot_df.copy()
    label_df["label_priority"] = label_df["family"].apply(
        lambda x: 0 if x == "APS-LRGR" else 1
    )
    label_df = label_df.sort_values(["label_priority", "family", x_col])

    for _, row in label_df.iterrows():
        family = row["family"]

        if label_mode == "none":
            continue
        if label_mode == "aps_only" and family != "APS-LRGR":
            continue

        label = short_label(row["model_name"])

        add_non_overlapping_label(
            fig=fig,
            ax=ax,
            x=row[x_col],
            y=row[ACC_COL],
            text=label,
            used_bboxes=used_bboxes,
            family=family,
        )

    # 凡例
    handles = []
    for family, color in family_colors.items():
        if family in plot_df["family"].unique():
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=family_markers.get(family, "o"),
                    color="w",
                    label=family,
                    markerfacecolor=color,
                    markeredgecolor="black",
                    markersize=12,
                )
            )

    ax.legend(
        handles=handles,
        title="Model family",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
        borderpad=0.8,
        labelspacing=0.6,
    )

    fig.tight_layout()

    save_path_png = os.path.join(out_dir, filename + "_slide.png")
    save_path_pdf = os.path.join(out_dir, filename + "_slide.pdf")

    fig.savefig(save_path_png, dpi=300, bbox_inches="tight", pad_inches=0.25)
    fig.savefig(save_path_pdf, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)

    print(f"saved: {save_path_png}")
    print(f"saved: {save_path_pdf}")

# =========================================================
# 10. Generate figures
# =========================================================
if __name__ == "__main__":
    print("summary:", summary)
    print("output:", out_dir)
    print()
    print("models used:")
    for name in df["model_name"].tolist():
        print(" -", name)
    print()

    plot_tradeoff_slide(
        df,
        x_col="params_trainable",
        x_label="Trainable Parameters",
        filename="accuracy_vs_trainable_params",
        log_x=True,
        label_mode="all",
    )

    plot_tradeoff_slide(
        df,
        x_col="flops_m",
        x_label="FLOPs (M)",
        filename="accuracy_vs_flops",
        log_x=True,
        label_mode="all",
    )

    plot_tradeoff_slide(
        df,
        x_col="latency_sample_ms",
        x_label="Latency per Sample (ms)",
        filename="accuracy_vs_latency",
        log_x=True,
        label_mode="all",
    )

    plot_tradeoff_slide(
        df,
        x_col="size_mb",
        x_label="Model Size (MB)",
        filename="accuracy_vs_model_size",
        log_x=True,
        label_mode="all",
    )