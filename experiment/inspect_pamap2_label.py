import numpy as np

X = np.load("../dataset/pamap2_100hz_w496_s248_chest16g/X.npy")
Y = np.load("../dataset/pamap2_100hz_w496_s248_chest16g/Y.npy")
G = np.load("../dataset/pamap2_100hz_w496_s248_chest16g/groups.npy")

print("[X]", X.shape, X.dtype)
print("[Y]", Y.shape, Y.dtype)
print("[G]", G.shape, G.dtype)

assert X.shape[0] == Y.shape[0] == G.shape[0], "X/Y/G length mismatch"

# 109除外（あなたの方針）
mask = (G != 109)
X, Y, G = X[mask], Y[mask], G[mask]
print("[AFTER mask!=109] N =", len(Y), "subjects =", np.unique(G))

# ラベルの基本統計
uniq, cnt = np.unique(Y, return_counts=True)
print("[LABEL unique]", uniq.tolist())
print("[LABEL counts]")
for u, c in zip(uniq, cnt):
    print(" ", int(u), int(c), f"ratio={c/len(Y):.4f}")

# 重要：0が残ってたら、そのまま学習・評価に使うのは危険
num_zero = int((Y == 0).sum())
print("[Y==0] count =", num_zero, "ratio =", num_zero/len(Y))

# NaN/Inf
print("[X nan]", int(np.isnan(X).sum()), "[X inf]", int(np.isinf(X).sum()))
print("[Y nan]", int(np.isnan(Y).sum()) if np.issubdtype(Y.dtype, np.floating) else 0)

# 被験者ごとのラベル種類（LOSOで偏りがないか）
print("\n[PER SUBJECT label variety]")
for s in np.unique(G):
    ys = Y[G == s]
    u = np.unique(ys)
    print(" subject", int(s), "windows", ys.size, "unique_labels", u.tolist())
