import argparse
import os
import numpy as np

def load_dat(path, max_rows=None):
    kwargs = {}
    if max_rows is not None:
        kwargs["max_rows"] = max_rows
    x = np.genfromtxt(path, delimiter=" ", dtype=np.float32, invalid_raise=False, **kwargs)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--subject", type=str, default="subject109")
    ap.add_argument("--session", type=str, default="Protocol", choices=["Protocol", "Optional"])
    ap.add_argument("--peek_rows", type=int, default=3)
    ap.add_argument("--max_rows", type=int, default=200000)
    args = ap.parse_args()

    path = os.path.join(args.root, args.session, f"{args.subject}.dat")
    print("[PATH]", path)
    x = load_dat(path, max_rows=args.max_rows)

    print("[SHAPE]", x.shape, "dtype=", x.dtype)
    print("[COLUMNS] expected=54, actual=", x.shape[1])

    print("[HEAD]")
    for i in range(min(args.peek_rows, x.shape[0])):
        print("  ", x[i].tolist())

    ts = x[:, 0]
    act = x[:, 1]
    hr = x[:, 2]

    print("[TIMESTAMP] example:", ts[:10].tolist())
    print("[ACTIVITY] unique IDs (first pass):", np.unique(act[~np.isnan(act)]).astype(int).tolist()[:50])
    print("[ACTIVITY] count act=0:", int(np.sum(act == 0)), "/", x.shape[0])

    print("[NAN CHECK]")
    print("  hr_nan:", int(np.isnan(hr).sum()), "/", hr.shape[0])
    print("  any_nan_in_rows:", int(np.isnan(x).any(axis=1).sum()), "/", x.shape[0])

    chest_acc16 = x[:, 21:24]
    print("[CHEST_ACC_16G] shape:", chest_acc16.shape)
    print("[CHEST_ACC_16G] example first row:", chest_acc16[0].tolist())
    print("[CHEST_ACC_16G] nan_rows:", int(np.isnan(chest_acc16).any(axis=1).sum()), "/", chest_acc16.shape[0])

if __name__ == "__main__":
    main()
