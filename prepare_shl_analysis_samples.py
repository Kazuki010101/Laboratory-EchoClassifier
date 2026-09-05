import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from datasets import CustomTransform


CLASS_NAMES = [
    "Still",
    "Walking",
    "Run",
    "Bike",
    "Car",
    "Bus",
    "Train",
    "Subway",
]


DATA_ROOT = Path(
    "dataset/2023_processed/"
    "100hz_5.0s_overlap0.0s"
)

ACC_PATH = DATA_ROOT / "Hips_Acc.npy"
LABEL_PATH = DATA_ROOT / "Hips_Label.npy"

OUTPUT_ROOT = Path(
    "experiments/shl2023_senvt_kd/"
    "data_splits"
)

SAMPLES_PER_CLASS = 5
INPUT_SIZE = 496
SPLIT_RANDOM_STATE = 0


def main():
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not ACC_PATH.is_file():
        raise FileNotFoundError(
            f"Missing: {ACC_PATH}"
        )

    if not LABEL_PATH.is_file():
        raise FileNotFoundError(
            f"Missing: {LABEL_PATH}"
        )

    acceleration = np.load(
        ACC_PATH,
        mmap_mode="r",
    )

    raw_labels = np.load(
        LABEL_PATH,
        mmap_mode="r",
    )

    labels = np.asarray(
        raw_labels,
        dtype=np.int64,
    ) - 1

    if len(acceleration) != len(labels):
        raise RuntimeError(
            "Acceleration and label counts differ."
        )

    if acceleration.ndim != 3:
        raise RuntimeError(
            "Expected acceleration shape [N, 3, L], "
            f"got {acceleration.shape}."
        )

    if set(np.unique(labels).tolist()) != set(range(8)):
        raise RuntimeError(
            "Expected model labels 0 through 7."
        )

    all_indices = np.arange(
        len(labels),
        dtype=np.int64,
    )

    (
        _,
        _,
        train_indices,
        validation_indices,
    ) = train_test_split(
        labels,
        all_indices,
        train_size=0.9,
        random_state=SPLIT_RANDOM_STATE,
    )

    overlap = np.intersect1d(
        train_indices,
        validation_indices,
    )

    if len(overlap) != 0:
        raise RuntimeError(
            "Train and validation indices overlap."
        )

    split_path = (
        OUTPUT_ROOT
        / "shl2023_distillation_split_indices.npz"
    )

    np.savez_compressed(
        split_path,
        train_indices=train_indices,
        validation_indices=validation_indices,
        random_state=np.asarray(
            SPLIT_RANDOM_STATE,
            dtype=np.int64,
        ),
    )

    selected_records = []
    selected_global_indices = []
    selected_validation_positions = []
    selected_labels = []

    class_counts = {
        label: 0
        for label in range(8)
    }

    for validation_position, global_index in enumerate(
        validation_indices
    ):
        label = int(
            labels[global_index]
        )

        if class_counts[label] >= SAMPLES_PER_CLASS:
            continue

        selected_records.append(
            {
                "analysis_sample_index": len(
                    selected_records
                ),
                "global_index": int(
                    global_index
                ),
                "validation_position": int(
                    validation_position
                ),
                "label": label,
                "class_name": CLASS_NAMES[label],
            }
        )

        selected_global_indices.append(
            int(global_index)
        )

        selected_validation_positions.append(
            int(validation_position)
        )

        selected_labels.append(
            label
        )

        class_counts[label] += 1

        if all(
            count >= SAMPLES_PER_CLASS
            for count in class_counts.values()
        ):
            break

    if not all(
        count == SAMPLES_PER_CLASS
        for count in class_counts.values()
    ):
        raise RuntimeError(
            "Could not select the requested number "
            "of samples for every class."
        )

    deterministic_transform = CustomTransform(
        signal_length=INPUT_SIZE,
        is_training=False,
    )

    transformed_samples = []

    for global_index in selected_global_indices:
        raw_sample = np.asarray(
            acceleration[global_index],
            dtype=np.float32,
        )

        transformed_sample = (
            deterministic_transform(
                raw_sample
            )
        )

        transformed_samples.append(
            transformed_sample.numpy()
        )

    transformed_samples = np.stack(
        transformed_samples,
        axis=0,
    ).astype(np.float32)

    analysis_samples_path = (
        OUTPUT_ROOT
        / "shl2023_analysis_samples.npz"
    )

    np.savez_compressed(
        analysis_samples_path,
        samples=transformed_samples,
        labels=np.asarray(
            selected_labels,
            dtype=np.int64,
        ),
        global_indices=np.asarray(
            selected_global_indices,
            dtype=np.int64,
        ),
        validation_positions=np.asarray(
            selected_validation_positions,
            dtype=np.int64,
        ),
    )

    manifest = {
        "dataset": "SHL2023",
        "sensor_position": "Hips",
        "sensor": "Acc",
        "source_shape": list(
            acceleration.shape
        ),
        "source_dtype": str(
            acceleration.dtype
        ),
        "input_size": INPUT_SIZE,
        "split_train_size": 0.9,
        "split_random_state": (
            SPLIT_RANDOM_STATE
        ),
        "train_count": int(
            len(train_indices)
        ),
        "validation_count": int(
            len(validation_indices)
        ),
        "samples_per_class": (
            SAMPLES_PER_CLASS
        ),
        "samples": selected_records,
    }

    manifest_path = (
        OUTPUT_ROOT
        / "shl2023_analysis_sample_manifest.json"
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            manifest,
            output_file,
            ensure_ascii=False,
            indent=2,
        )

    first_raw = np.asarray(
        acceleration[
            selected_global_indices[0]
        ],
        dtype=np.float32,
    )

    first_a = deterministic_transform(
        first_raw
    )

    first_b = deterministic_transform(
        first_raw
    )

    print("=== SHL ANALYSIS DATA ===")
    print("source_shape:", acceleration.shape)
    print("source_dtype:", acceleration.dtype)
    print("label_range:", labels.min(), labels.max())
    print("train_count:", len(train_indices))
    print(
        "validation_count:",
        len(validation_indices),
    )
    print(
        "train_validation_overlap:",
        len(overlap),
    )
    print(
        "analysis_samples_shape:",
        transformed_samples.shape,
    )
    print(
        "analysis_labels:",
        selected_labels,
    )
    print(
        "transformed_shape:",
        tuple(first_a.shape),
    )
    print(
        "deterministic_transform:",
        torch_equal(first_a, first_b),
    )
    print("split:", split_path)
    print("samples:", analysis_samples_path)
    print("manifest:", manifest_path)


def torch_equal(first, second):
    return bool(
        np.array_equal(
            first.numpy(),
            second.numpy(),
        )
    )


if __name__ == "__main__":
    main()
