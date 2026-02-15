import numpy as np
from collections import Counter

ACC_PATH = "SHL_2023/SHL/100hz_5.0s_overlap0.0s/SHL/train/Hips_Acc.npy"
LAB_PATH = "SHL_2023/SHL/100hz_5.0s_overlap0.0s/SHL/train/Hips_Label.npy"

HEAD_N = 30
TAIL_N = 30
ACC_SAMPLE_WINDOWS = 5000
TOPK = 20


def load_npy(path: str):
    arr = np.load(path, mmap_mode="r", allow_pickle=False)
    return arr


def head_tail_1d(x: np.ndarray, k_head: int, k_tail: int):
    n = int(x.shape[0])
    h = np.asarray(x[: min(k_head, n)])
    t = np.asarray(x[max(0, n - k_tail): n])
    return h, t


def numeric_stats(arr: np.ndarray, sample_rows: int | None = None):
    a = arr
    if sample_rows is not None and a.ndim >= 1 and a.shape[0] > sample_rows:
        a = a[:sample_rows]
    a = np.asarray(a, dtype=np.float64)
    finite = np.isfinite(a)
    if not np.any(finite):
        return None
    v = a[finite]
    return {
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "finite_ratio": float(v.size) / float(a.size),
    }


def count_runs_1d(x: np.ndarray):
    x = np.asarray(x)
    if x.size == 0:
        return 0, 0.0
    if x.size == 1:
        return 1, 1.0
    changes = np.sum(x[1:] != x[:-1])
    runs = int(changes) + 1
    avg_run_len = float(x.size) / float(runs)
    return runs, avg_run_len


def looks_integer(x: np.ndarray):
    if np.issubdtype(x.dtype, np.integer):
        return True
    if np.issubdtype(x.dtype, np.floating):
        s = np.asarray(x[: min(x.shape[0], 2000)], dtype=np.float64)
        if s.size == 0:
            return False
        s = s[np.isfinite(s)]
        if s.size == 0:
            return False
        return bool(np.all(np.abs(s - np.round(s)) < 1e-6))
    return False


def unique_summary_1d(x: np.ndarray, topk: int):
    x = np.asarray(x)
    n = int(x.shape[0])
    if n == 0:
        return {"n_unique": 0, "top": []}
    if n > 2_000_000:
        xs = np.asarray(x[:2_000_000])
    else:
        xs = x
    try:
        vals, counts = np.unique(xs, return_counts=True)
        idx = np.argsort(-counts)
        top = [(vals[i].item() if hasattr(vals[i], "item") else vals[i], int(counts[i])) for i in idx[:topk]]
        return {"n_unique": int(vals.shape[0]), "top": top}
    except Exception:
        c = Counter(np.asarray(xs).tolist())
        top = c.most_common(topk)
        return {"n_unique": len(c), "top": top}


def id_candidate_score(x: np.ndarray):
    x = np.asarray(x)
    if x.ndim != 1 or x.size == 0:
        return -1.0
    if not looks_integer(x):
        return -1.0
    u = unique_summary_1d(x, topk=5)["n_unique"]
    runs, avg_run = count_runs_1d(x)
    n = x.size
    if u <= 1:
        return -1.0
    u_score = 0.0
    if 2 <= u <= 5000:
        u_score = 1.0
    elif 5000 < u <= 50000:
        u_score = 0.3
    else:
        u_score = -0.5
    run_score = min(1.0, avg_run / 50.0)
    change_rate = float(runs) / float(n)
    change_penalty = min(1.0, change_rate * 200.0)
    return 2.0 * u_score + 2.0 * run_score - 1.5 * change_penalty


def print_block(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def inspect_array(arr: np.ndarray, name: str):
    print_block(f"{name}: basic info")
    print(f"shape = {tuple(arr.shape)}")
    print(f"dtype = {arr.dtype}")

    if np.issubdtype(arr.dtype, np.number):
        s = numeric_stats(arr, sample_rows=ACC_SAMPLE_WINDOWS if arr.ndim >= 1 else None)
        if s is not None:
            print("numeric stats (sampled if large)")
            for k, v in s.items():
                print(f"  {k}: {v}")

    if arr.ndim == 1:
        x = arr
        h, t = head_tail_1d(x, HEAD_N, TAIL_N)
        print_block(f"{name}: contents (1D)")
        print(f"head({len(h)}): {h.tolist()}")
        print(f"tail({len(t)}): {t.tolist()}")
        us = unique_summary_1d(x, TOPK)
        runs, avg_run = count_runs_1d(np.asarray(x))
        print(f"n_unique: {us['n_unique']}")
        print(f"runs: {runs}, avg_run_len: {avg_run:.2f}")
        print("top frequencies:")
        for v, c in us["top"]:
            print(f"  {v}: {c}")
        sc = id_candidate_score(np.asarray(x))
        print(f"id_candidate_score: {sc:.3f}")

    elif arr.ndim == 2:
        print_block(f"{name}: contents (2D)")
        r = min(10, arr.shape[0])
        c = min(10, arr.shape[1])
        sample = np.asarray(arr[:r, :c])
        print(f"top-left sample ({r}x{c}):")
        print(sample)

        print_block(f"{name}: per-column ID candidate check")
        col_scores = []
        for j in range(arr.shape[1]):
            col = np.asarray(arr[:, j])
            sc = id_candidate_score(col)
            col_scores.append((j, sc))
        col_scores.sort(key=lambda x: x[1], reverse=True)

        shown = 0
        for j, sc in col_scores[: min(10, len(col_scores))]:
            col = np.asarray(arr[:, j])
            us = unique_summary_1d(col, 10)
            runs, avg_run = count_runs_1d(col)
            h, t = head_tail_1d(col, 15, 15)
            print(f"col={j} score={sc:.3f} dtype={col.dtype} n_unique={us['n_unique']} runs={runs} avg_run={avg_run:.2f}")
            print(f"  head: {h.tolist()}")
            print(f"  tail: {t.tolist()}")
            print("  top freq:")
            for v, c in us["top"]:
                print(f"    {v}: {c}")
            shown += 1
            if shown >= 3:
                break

    else:
        print_block(f"{name}: contents (ndim={arr.ndim})")
        first = np.asarray(arr[0])
        print(f"first element shape: {tuple(first.shape)}")
        print(f"first element sample:")
        flat = np.asarray(first).ravel()
        print(flat[: min(50, flat.size)].tolist())


def main():
    print_block("Loading npy files")
    acc = load_npy(ACC_PATH)
    lab = load_npy(LAB_PATH)
    print(f"ACC_PATH: {ACC_PATH}")
    print(f"LAB_PATH: {LAB_PATH}")

    inspect_array(acc, "ACC")
    inspect_array(lab, "LABEL")

    print_block("Consistency checks")
    if acc.ndim >= 1 and lab.ndim >= 1:
        n_acc = int(acc.shape[0])
        n_lab = int(lab.shape[0])
        print(f"n_windows in ACC: {n_acc}")
        print(f"n_rows in LABEL: {n_lab}")
        if n_acc == n_lab:
            print("OK: ACCとLABELの先頭次元が一致しています（ウィンドウ数一致）")
        else:
            print("NG: ACCとLABELの先頭次元が一致していません（対応付け要確認）")

    print_block("How to interpret for LOSO")
    if lab.ndim == 1:
        print("LABELが1次元です。通常はクラスラベルのみの可能性が高く、被験者IDはここには入っていないかもしれません。")
    elif lab.ndim == 2:
        print("LABELが2次元です。列のどれかに被験者ID（subject/session）が入っている可能性があります。上のスコア上位列を確認してください。")
    else:
        print("LABELの次元が想定外です。中身のフォーマットを要確認です。")


if __name__ == "__main__":
    main()
