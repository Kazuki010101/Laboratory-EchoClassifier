import argparse
from pathlib import Path
import numpy as np


def _count_lines_fast(path: Path) -> int:
    cnt = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            cnt += chunk.count(b"\n")
    return cnt


def _safe_head(path: Path, n: int) -> str:
    lines = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for _ in range(n):
            line = f.readline()
            if not line:
                break
            lines.append(line.rstrip("\n"))
    return "\n".join(lines)


def _load_txt(path: Path, dtype=None):
    try:
        return np.loadtxt(str(path), dtype=dtype)
    except Exception:
        return np.genfromtxt(str(path), dtype=dtype)


def _summarize_array(arr: np.ndarray, name: str):
    out = []
    out.append(f"  - name: {name}")
    out.append(f"  - dtype: {arr.dtype}")
    out.append(f"  - shape: {arr.shape}")

    if arr.size == 0:
        out.append("  - empty: True")
        return "\n".join(out)

    if arr.dtype.kind in ("f", "c"):
        nan_cnt = int(np.isnan(arr).sum())
        inf_cnt = int(np.isinf(arr).sum())
        out.append(f"  - nan_count: {nan_cnt}")
        out.append(f"  - inf_count: {inf_cnt}")
        finite = arr[np.isfinite(arr)]
        if finite.size > 0:
            out.append(f"  - min: {np.min(finite)}")
            out.append(f"  - max: {np.max(finite)}")
            out.append(f"  - mean: {np.mean(finite)}")
            out.append(f"  - std: {np.std(finite)}")
        else:
            out.append("  - all values non-finite")
    else:
        out.append(f"  - min: {np.min(arr)}")
        out.append(f"  - max: {np.max(arr)}")
        out.append(f"  - unique_count: {np.unique(arr).shape[0] if arr.size < 5_000_000 else 'skip(too_large)'}")

    flat = arr.reshape(-1)
    preview = flat[: min(10, flat.shape[0])]
    out.append(f"  - first_values(<=10): {preview.tolist()}")
    return "\n".join(out)


def _resolve_paths(root: Path, split: str):
    if split == "train":
        acc = root / "train" / "Hips" / "Acc.txt"
        lab = root / "train" / "Hips" / "Label.txt"
    elif split in ("validate", "valid", "val"):
        acc = root / "validate" / "Hips" / "Acc.txt"
        lab = root / "validate" / "Hips" / "Label.txt"
        if not acc.exists():
            acc = root / "valid" / "Hips" / "Acc.txt"
            lab = root / "valid" / "Hips" / "Label.txt"
        if not acc.exists():
            acc = root / "val" / "Hips" / "Acc.txt"
            lab = root / "val" / "Hips" / "Label.txt"
    elif split == "test":
        acc = root / "test" / "Acc.txt"
        lab = root / "test" / "Label_idx.txt"
    else:
        raise ValueError(split)
    return acc, lab


def _inspect_pair(acc_path: Path, lab_path: Path, head_lines: int):
    print(f"\n[ACC] {acc_path}")
    if not acc_path.exists():
        print("  - exists: False")
        return None, None
    print("  - exists: True")
    print(f"  - size_bytes: {acc_path.stat().st_size}")
    print(f"  - line_count: {_count_lines_fast(acc_path)}")
    print("  - head:")
    print(_safe_head(acc_path, head_lines))
    acc = _load_txt(acc_path, dtype=np.float32)
    if isinstance(acc, np.ndarray) and acc.ndim == 0:
        acc = np.array([acc], dtype=np.float32)
    print(_summarize_array(acc, acc_path.name))

    print(f"\n[LABEL] {lab_path}")
    if not lab_path.exists():
        print("  - exists: False")
        return acc, None
    print("  - exists: True")
    print(f"  - size_bytes: {lab_path.stat().st_size}")
    print(f"  - line_count: {_count_lines_fast(lab_path)}")
    print("  - head:")
    print(_safe_head(lab_path, head_lines))

    lab = _load_txt(lab_path)
    if isinstance(lab, np.ndarray) and lab.ndim == 0:
        lab = np.array([lab])
    print(_summarize_array(lab, lab_path.name))

    print("\n[ALIGNMENT CHECK]")
    acc_n = acc.shape[0] if isinstance(acc, np.ndarray) and acc.ndim >= 1 else None
    lab_n = lab.shape[0] if isinstance(lab, np.ndarray) and lab.ndim >= 1 else None
    print(f"  - acc_samples (acc.shape[0]): {acc_n}")
    print(f"  - label_samples (label.shape[0]): {lab_n}")

    if acc_n is not None and lab_n is not None:
        if acc_n == lab_n:
            print("  - match: True")
        else:
            print("  - match: False")
            print("  - note: AccとLabelの1行=1サンプル対応になっていない可能性あり")

    return acc, lab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="dataset/SHL_2023")
    parser.add_argument("--head_lines", type=int, default=5)
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"root not found: {root}")

    print(f"[ROOT] {root.resolve()}")
    for split in ["train", "validate", "test"]:
        acc_path, lab_path = _resolve_paths(root, split)
        print(f"\n==================== SPLIT: {split} ====================")
        _inspect_pair(acc_path, lab_path, args.head_lines)

    print("\n[DONE]")


if __name__ == "__main__":
    main()
