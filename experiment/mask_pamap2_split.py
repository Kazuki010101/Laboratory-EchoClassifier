import numpy as np

X = np.load("../dataset/pamap2_100hz_w496_s248_chest16g/X.npy")
Y = np.load("../dataset/pamap2_100hz_w496_s248_chest16g/Y.npy")
G = np.load("../dataset/pamap2_100hz_w496_s248_chest16g/groups.npy")

mask = (G != 109)
X, Y, G = X[mask], Y[mask], G[mask]

subjects = np.unique(G)

splits = []
for i, test_subj in enumerate(subjects):
    rest = subjects[subjects != test_subj]
    val_subj = rest[i % len(rest)]

    train_subjs = rest[rest != val_subj]

    train_idx = np.isin(G, train_subjs)
    val_idx   = (G == val_subj)
    test_idx  = (G == test_subj)

    splits.append({
        "test_subj": int(test_subj),
        "val_subj": int(val_subj),
        "n_train": int(train_idx.sum()),
        "n_val": int(val_idx.sum()),
        "n_test": int(test_idx.sum()),
        "train_idx": np.where(train_idx)[0],
        "val_idx": np.where(val_idx)[0],
        "test_idx": np.where(test_idx)[0],
    })

for s in splits[:8]:
    print(s["test_subj"], "train/val/test =", s["n_train"], s["n_val"], s["n_test"])
