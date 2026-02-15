import argparse
import os
import re
import json
import numpy as np

ACTIVITY_NAME = {
    0: "other/transient(discard)",
    1: "lying",
    2: "sitting",
    3: "standing",
    4: "walking",
    5: "running",
    6: "cycling",
    7: "Nordic walking",
    9: "watching TV",
    10: "computer work",
    11: "car driving",
    12: "ascending stairs",
    13: "descending stairs",
    16: "vacuum cleaning",
    17: "ironing",
    18: "folding laundry",
    19: "house cleaning",
    20: "playing soccer",
    24: "rope jumping",
}

def _parse_subject_id(subject_str: str) -> int:
    m = re.search(r"(\d+)", subject_str)
    if not m:
        raise ValueError(f"Invalid subject: {subject_str}")
    return int(m.group(1))

def _load_dat(path: str) -> np.ndarray:
    data = np.loadtxt(path, delimiter=" ", dtype=np.float32)
    if data.ndim != 2 or data.shape[1] != 54:
        raise ValueError(f"Unexpected shape: {data.shape} in {path}")
    return data

def _majority_label(y: np.ndarray) -> int:
    vals, cnts = np.unique(y, return_counts=True)
    return int(vals[np.argmax(cnts)])

def _make_segments(t: np.ndarray, max_dt: float = 0.011) -> np.ndarray:
    dt = np.diff(t)
    cut = np.where(dt > max_dt)[0]
    starts = np.concatenate([[0], cut + 1])
    ends = np.concatenate([cut + 1, [len(t)]])
    return np.stack([starts, ends], axis=1)

def _windowize(x: np.ndarray, y: np.ndarray, t: np.ndarray, win: int, step: int):
    segments = _make_segments(t)
    Xw = []
    Yw = []
    for s, e in segments:
        L = e - s
        if L < win:
            continue
        for i in range(s, e - win + 1, step):
            xx = x[i:i + win]
            yy = y[i:i + win]
            Xw.append(xx)
            Yw.append(_majority_label(yy))
    if len(Xw) == 0:
        return np.empty((0, win, x.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.int64)
    Xw = np.stack(Xw, axis=0).astype(np.float32)
    Yw = np.array(Yw, dtype=np.int64)
    return Xw, Yw

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--session", default="Protocol", choices=["Protocol", "Optional"])
    ap.add_argument("--sensor", default="chest", choices=["hand", "chest", "ankle"])
    ap.add_argument("--acc", default="16g", choices=["16g", "6g"])
    ap.add_argument("--window", type=int, default=496)
    ap.add_argument("--step", type=int, default=248)
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    base = os.path.join(args.root, args.session)
    if not os.path.isdir(base):
        raise FileNotFoundError(base)

    if args.subjects is None:
        files = sorted([f for f in os.listdir(base) if f.startswith("subject") and f.endswith(".dat")])
        subjects = [os.path.splitext(f)[0] for f in files]
    else:
        subjects = args.subjects

    if args.sensor == "hand":
        if args.acc == "16g":
            cols = (5, 6, 7)
        else:
            cols = (8, 9, 10)
    elif args.sensor == "chest":
        if args.acc == "16g":
            cols = (22, 23, 24)
        else:
            cols = (25, 26, 27)
    else:
        if args.acc == "16g":
            cols = (39, 40, 41)
        else:
            cols = (42, 43, 44)

    all_X = []
    all_Y = []
    all_G = []
    activity_counter = {}

    for subj in subjects:
        sid = _parse_subject_id(subj)
        path = os.path.join(base, f"{subj}.dat")
        data = _load_dat(path)

        t = data[:, 0]
        act = data[:, 1].astype(np.int64)

        x = data[:, [c - 1 for c in cols]]

        valid = (act != 0)
        valid = valid & np.isfinite(x).all(axis=1)
        valid = valid & np.isfinite(t)

        t = t[valid]
        act = act[valid]
        x = x[valid]

        for a, c in zip(*np.unique(act, return_counts=True)):
            activity_counter[int(a)] = activity_counter.get(int(a), 0) + int(c)

        Xw, Yw = _windowize(x, act, t, win=args.window, step=args.step)
        Gw = np.full((Yw.shape[0],), sid, dtype=np.int16)

        all_X.append(Xw)
        all_Y.append(Yw)
        all_G.append(Gw)

        print(f"[{subj}] rows={len(act)} windows={len(Yw)} sensor={args.sensor} acc={args.acc} cols={cols}")

    X = np.concatenate(all_X, axis=0) if len(all_X) else np.empty((0, args.window, 3), dtype=np.float32)
    Y = np.concatenate(all_Y, axis=0) if len(all_Y) else np.empty((0,), dtype=np.int64)
    G = np.concatenate(all_G, axis=0) if len(all_G) else np.empty((0,), dtype=np.int16)

    X = np.transpose(X, (0, 2, 1)).astype(np.float32)

    np.save(os.path.join(args.out_dir, "X.npy"), X)
    np.save(os.path.join(args.out_dir, "Y.npy"), Y)
    np.save(os.path.join(args.out_dir, "groups.npy"), G)

    meta = {
        "root": args.root,
        "session": args.session,
        "sensor": args.sensor,
        "acc": args.acc,
        "cols_1based": list(cols),
        "window": args.window,
        "step": args.step,
        "X_shape": list(X.shape),
        "Y_shape": list(Y.shape),
        "groups_shape": list(G.shape),
        "activity_distribution_rows": {
            str(k): {"count": int(v), "name": ACTIVITY_NAME.get(k, "unknown")}
            for k, v in sorted(activity_counter.items(), key=lambda kv: kv[0])
        },
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[SAVED] {args.out_dir}/X.npy, Y.npy, groups.npy, meta.json")
    print(f"[X] {X.shape} float32, [Y] {Y.shape} int64, [groups] {G.shape} int16")

if __name__ == "__main__":
    main()
