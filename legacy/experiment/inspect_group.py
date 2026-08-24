import os
import numpy as np

def main():
    root = "../dataset/pamap2_100hz_w496_s248_chest16g"
    path = os.path.join(root, "groups.npy")

    g = np.load(path)

    print("[PATH]", path)
    print("[DTYPE]", g.dtype)
    print("[SHAPE]", g.shape)

    uniq, counts = np.unique(g, return_counts=True)
    order = np.argsort(uniq)
    uniq, counts = uniq[order], counts[order]

    print("\n[UNIQUE SUBJECTS / COUNTS]")
    for u, c in zip(uniq.tolist(), counts.tolist()):
        print(f"  subject{int(u)}: {c}")

    print("\n[HEAD 50]")
    print(g[:50].tolist())

    print("\n[TAIL 50]")
    print(g[-50:].tolist())

    changes = np.where(g[1:] != g[:-1])[0]
    print("\n[CHANGES] number_of_boundaries =", int(changes.size))
    if changes.size > 0:
        show = changes[:10]
        for i in show.tolist():
            print(f"  idx {i} -> {i+1}: {int(g[i])} -> {int(g[i+1])}")
        if changes.size > 10:
            print("  ...")

if __name__ == "__main__":
    main()
