import json
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


DEFAULT_CLASS_NAMES = [
    "Still",
    "Walking",
    "Run",
    "Bike",
    "Car",
    "Bus",
    "Train",
    "Subway",
]


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()

    return np.asarray(value)


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(
        logits,
        dtype=np.float64,
    )

    logits = logits - np.max(logits)

    probabilities = np.exp(logits)

    return probabilities / np.sum(probabilities)


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = np.maximum(values, 0.0)

    total = values.sum()

    if total <= 0:
        return np.full_like(
            values,
            1.0 / len(values),
        )

    return values / total


def save_patch_selection_report(
    input_signal,
    true_label: int,
    teacher_logits,
    student_logits,
    teacher_time_importance,
    teacher_patch_importance,
    aps_patch_probabilities,
    selected_indices,
    patch_size: int,
    stride: int,
    output_dir,
    sample_index: int,
    class_names: Optional[Sequence[str]] = None,
):
    """
    Save one-sample teacher/APS analysis as PNG and JSON.

    Args:
        input_signal:
            [C, L]

        teacher_logits:
            [num_classes]

        student_logits:
            [num_classes]

        teacher_time_importance:
            [L]

        teacher_patch_importance:
            [num_patches]

        aps_patch_probabilities:
            [num_patches]

        selected_indices:
            [num_selected_patches]
    """

    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    input_signal = _to_numpy(
        input_signal
    ).astype(np.float64)

    teacher_logits = _to_numpy(
        teacher_logits
    ).reshape(-1)

    student_logits = _to_numpy(
        student_logits
    ).reshape(-1)

    teacher_time_importance = _to_numpy(
        teacher_time_importance
    ).reshape(-1)

    teacher_patch_importance = _normalize(
        _to_numpy(
            teacher_patch_importance
        ).reshape(-1)
    )

    aps_patch_probabilities_raw = _to_numpy(
        aps_patch_probabilities
    ).reshape(-1)

    aps_patch_distribution = _normalize(
        aps_patch_probabilities_raw
    )

    selected_indices = _to_numpy(
        selected_indices
    ).astype(np.int64).reshape(-1)

    if input_signal.ndim != 2:
        raise ValueError(
            "input_signal must have shape [C, L], "
            f"got {input_signal.shape}."
        )

    channel_count, input_length = (
        input_signal.shape
    )

    patch_count = len(
        teacher_patch_importance
    )

    if len(aps_patch_probabilities_raw) != patch_count:
        raise ValueError(
            "Teacher and APS patch counts do not match: "
            f"{patch_count} vs "
            f"{len(aps_patch_probabilities_raw)}."
        )

    if len(teacher_time_importance) != input_length:
        raise ValueError(
            "Teacher time importance length does not "
            "match the input length."
        )

    if len(class_names) != len(teacher_logits):
        raise ValueError(
            "class_names and teacher logits do not match."
        )

    if len(student_logits) != len(teacher_logits):
        raise ValueError(
            "Teacher and student class counts do not match."
        )

    teacher_probabilities = _softmax(
        teacher_logits
    )

    student_probabilities = _softmax(
        student_logits
    )

    teacher_prediction = int(
        teacher_probabilities.argmax()
    )

    student_prediction = int(
        student_probabilities.argmax()
    )

    selected_mask = np.zeros(
        patch_count,
        dtype=bool,
    )

    if len(selected_indices) > 0:
        if (
            selected_indices.min() < 0
            or selected_indices.max() >= patch_count
        ):
            raise ValueError(
                "selected_indices contains an invalid index."
            )

        selected_mask[selected_indices] = True

    time_axis = np.arange(
        input_length
    )

    patch_axis = np.arange(
        patch_count
    )

    figure, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(14, 14),
        constrained_layout=True,
    )

    # --------------------------------------------------
    # 1. Input signal and selected patch spans
    # --------------------------------------------------

    channel_labels = [
        "Acc-X",
        "Acc-Y",
        "Acc-Z",
    ]

    for channel_index in range(channel_count):
        if channel_index < len(channel_labels):
            label = channel_labels[channel_index]
        else:
            label = f"Channel-{channel_index}"

        axes[0].plot(
            time_axis,
            input_signal[channel_index],
            linewidth=1.0,
            label=label,
        )

    for patch_index in selected_indices:
        start = int(
            patch_index * stride
        )

        end = min(
            start + patch_size,
            input_length,
        )

        axes[0].axvspan(
            start,
            end,
            color="tab:orange",
            alpha=0.15,
        )

    axes[0].set_title(
        "Input acceleration and APS-selected patches"
    )

    axes[0].set_xlabel(
        "Time sample"
    )

    axes[0].set_ylabel(
        "Acceleration"
    )

    axes[0].legend(
        loc="upper right"
    )

    axes[0].grid(
        alpha=0.25
    )

    # --------------------------------------------------
    # 2. Teacher time importance
    # --------------------------------------------------

    axes[1].plot(
        time_axis,
        teacher_time_importance,
        color="tab:red",
        linewidth=1.2,
    )

    for patch_index in selected_indices:
        start = int(
            patch_index * stride
        )

        end = min(
            start + patch_size,
            input_length,
        )

        axes[1].axvspan(
            start,
            end,
            color="tab:orange",
            alpha=0.15,
        )

    axes[1].set_title(
        "SENvT teacher time importance"
    )

    axes[1].set_xlabel(
        "Time sample"
    )

    axes[1].set_ylabel(
        "Normalized importance"
    )

    axes[1].grid(
        alpha=0.25
    )

    # --------------------------------------------------
    # 3. Teacher patch importance vs APS scores
    # --------------------------------------------------

    bar_width = 0.4

    axes[2].bar(
        patch_axis - bar_width / 2,
        teacher_patch_importance,
        width=bar_width,
        color="tab:red",
        alpha=0.8,
        label="Teacher importance",
    )

    aps_colors = [
        "tab:orange" if selected else "tab:blue"
        for selected in selected_mask
    ]

    axes[2].bar(
        patch_axis + bar_width / 2,
        aps_patch_distribution,
        width=bar_width,
        color=aps_colors,
        alpha=0.8,
        label="APS score",
    )

    axes[2].set_title(
        "Teacher patch importance vs APS distribution"
    )

    axes[2].set_xlabel(
        "Patch index"
    )

    axes[2].set_ylabel(
        "Normalized score"
    )

    axes[2].legend(
        loc="upper right"
    )

    axes[2].grid(
        axis="y",
        alpha=0.25,
    )

    # --------------------------------------------------
    # 4. Teacher vs student class probabilities
    # --------------------------------------------------

    class_axis = np.arange(
        len(class_names)
    )

    axes[3].bar(
        class_axis - bar_width / 2,
        teacher_probabilities,
        width=bar_width,
        color="tab:red",
        alpha=0.8,
        label="SENvT teacher",
    )

    axes[3].bar(
        class_axis + bar_width / 2,
        student_probabilities,
        width=bar_width,
        color="tab:blue",
        alpha=0.8,
        label="PRC + APS",
    )

    axes[3].set_xticks(
        class_axis
    )

    axes[3].set_xticklabels(
        class_names,
        rotation=30,
        ha="right",
    )

    axes[3].set_ylim(
        0.0,
        1.0,
    )

    axes[3].set_ylabel(
        "Probability"
    )

    axes[3].set_title(
        "Class probabilities"
    )

    axes[3].legend(
        loc="upper right"
    )

    axes[3].grid(
        axis="y",
        alpha=0.25,
    )

    true_name = class_names[
        int(true_label)
    ]

    teacher_name = class_names[
        teacher_prediction
    ]

    student_name = class_names[
        student_prediction
    ]

    figure.suptitle(
        f"Sample {sample_index} | "
        f"True={true_name} | "
        f"Teacher={teacher_name} | "
        f"Student={student_name} | "
        f"Keep={len(selected_indices)}/{patch_count}",
        fontsize=14,
    )

    file_stem = (
        f"sample_{sample_index:06d}"
    )

    png_path = output_dir / (
        file_stem + ".png"
    )

    json_path = output_dir / (
        file_stem + ".json"
    )

    figure.savefig(
        png_path,
        dpi=200,
    )

    plt.close(
        figure
    )

    report = {
        "sample_index": int(sample_index),
        "true_label": int(true_label),
        "true_class": true_name,
        "teacher_prediction": teacher_prediction,
        "teacher_class": teacher_name,
        "student_prediction": student_prediction,
        "student_class": student_name,
        "patch_size": int(patch_size),
        "stride": int(stride),
        "total_patch_count": int(patch_count),
        "selected_patch_count": int(
            len(selected_indices)
        ),
        "keep_ratio": float(
            len(selected_indices) / patch_count
        ),
        "selected_indices": (
            selected_indices.tolist()
        ),
        "selected_mask": (
            selected_mask.astype(int).tolist()
        ),
        "teacher_logits": (
            teacher_logits.tolist()
        ),
        "student_logits": (
            student_logits.tolist()
        ),
        "teacher_probabilities": (
            teacher_probabilities.tolist()
        ),
        "student_probabilities": (
            student_probabilities.tolist()
        ),
        "teacher_time_importance": (
            teacher_time_importance.tolist()
        ),
        "teacher_patch_importance": (
            teacher_patch_importance.tolist()
        ),
        "aps_patch_probabilities_raw": (
            aps_patch_probabilities_raw.tolist()
        ),
        "aps_patch_distribution": (
            aps_patch_distribution.tolist()
        ),
    }

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            report,
            output_file,
            ensure_ascii=False,
            indent=2,
        )

    return {
        "png_path": str(png_path),
        "json_path": str(json_path),
    }
