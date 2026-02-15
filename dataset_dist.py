#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MOMENT-1-large を SHL-2023 で fine-tuning する最小実行スクリプト
（後で教師モデルとして使えるように、学習済み重みを保存）

依存:
    pip install torch torchvision timm momentfm scikit-learn scipy numpy
"""

import os, math, time, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from scipy import signal

# MOMENT
from momentfm.models.moment import MOMENT


# --------------------------
# データ前処理と Dataset
# --------------------------
class CustomTransform:
    def __init__(self, signal_length, is_training=True, jitter_strength=0.1, flip_prob=0.5):
        self.signal_length = signal_length
        self.is_training = is_training
        self.jitter_strength = jitter_strength
        self.flip_prob = flip_prob

    def __call__(self, x):
        y = np.zeros((x.shape[0], self.signal_length), dtype=np.float32)
        for i in range(x.shape[0]):
            y[i] = signal.resample(x[i], self.signal_length)
        y = torch.from_numpy(y).float()
        if self.is_training:
            y = y + torch.randn_like(y) * self.jitter_strength
            if random.random() < self.flip_prob:
                y = torch.flip(y, dims=[-1])
        return y


class ResampleToTensor:
    def __init__(self, signal_length):
        self.signal_length = signal_length
    def __call__(self, x):
        y = np.zeros((x.shape[0], self.signal_length), dtype=np.float32)
        for i in range(x.shape[0]):
            y[i] = signal.resample(x[i], self.signal_length)
        return torch.from_numpy(y).float()


class TriaxialSignalDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        x, y = self.data[idx], self.labels[idx]
        if self.transform: x = self.transform(x)
        return x, y


# --------------------------
# モデル定義
# --------------------------
class MOMENTForClassification(nn.Module):
    """MOMENT encoder + Linear head"""
    def __init__(self, moment_id, num_classes, input_size):
        super().__init__()
        self.backbone = MOMENT.from_pretrained(moment_id)
        hidden = getattr(self.backbone.config, "hidden_size", 512)
        self.head = nn.Linear(hidden, num_classes)

    def _forward_features(self, x):
        out = self.backbone(x)
        if isinstance(out, dict):
            for key in ["cls", "pooled", "features", "last_hidden_state"]:
                if key in out:
                    feat = out[key]; break
            else:
                feat = next(iter(out.values()))
        else:
            feat = out
        if feat.dim() == 3:
            feat = feat.mean(dim=1)
        return feat

    def forward(self, x):
        feat = self._forward_features(x)
        return self.head(feat)


# --------------------------
# 学習関数
# --------------------------
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    def forward(self, preds, target):
        log_probs = F.log_softmax(preds, dim=-1)
        nll = F.nll_loss(log_probs, target, reduction='none')
        smooth = -log_probs.mean(dim=-1)
        return ((1 - self.smoothing) * nll + self.smoothing * smooth).mean()


@torch.no_grad()
def accuracy(output, target):
    pred = output.argmax(dim=1)
    correct = pred.eq(target).sum().item()
    return 100.0 * correct / len(target)


def cosine_lr(optimizer, base_lr, warmup, total_epochs, epoch):
    if epoch < warmup:
        lr = base_lr * (epoch + 1) / warmup
    else:
        lr = 0.5 * base_lr * (1 + math.cos(math.pi * (epoch - warmup) / (total_epochs - warmup)))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


# --------------------------
# メイン
# --------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(0); np.random.seed(0); torch.manual_seed(0)

    # データ読み込み
    acc = np.load(args.acc_path)
    lab = np.load(args.lab_path)
    lab = np.array([l - 1 for l in lab])
    tr_idx, va_idx = train_test_split(np.arange(len(lab)), train_size=0.9, random_state=0)
    train_tf = CustomTransform(signal_length=args.input_size, is_training=True)
    val_tf = ResampleToTensor(signal_length=args.input_size)
    ds_tr = TriaxialSignalDataset(acc[tr_idx], lab[tr_idx], transform=train_tf)
    ds_va = TriaxialSignalDataset(acc[va_idx], lab[va_idx], transform=val_tf)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=4)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # モデル
    model = MOMENTForClassification(args.moment_id, args.num_classes, args.input_size).to(device)
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)

    best_acc = 0
    for ep in range(args.epochs):
        model.train(); total_loss = 0
        lr_now = cosine_lr(optimizer, args.lr, 5, args.epochs, ep)
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward(); optimizer.step()
            total_loss += loss.item() * x.size(0)
        train_loss = total_loss / len(ds_tr)

        # 評価
        model.eval(); total_acc = 0; total_loss = 0
        with torch.no_grad():
            for x, y in dl_va:
                x, y = x.to(device), y.to(device)
                out = model(x)
                total_loss += criterion(out, y).item() * x.size(0)
                total_acc += accuracy(out, y) * x.size(0)
        val_loss = total_loss / len(ds_va)
        val_acc = total_acc / len(ds_va)

        print(f"Epoch {ep+1}/{args.epochs}  lr={lr_now:.6f}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best_ckpt.pth"))
            print(f"  [saved best_ckpt.pth] acc={best_acc:.2f}%")

    print("done.")


# --------------------------
# CLI
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--acc-path", required=True, help="Hips_Acc.npy のパス")
    parser.add_argument("--lab-path", required=True, help="Hips_Label.npy のパス")
    parser.add_argument("--moment-id", default="AutonLab/MOMENT-1-large")
    parser.add_argument("--input-size", type=int, default=496)
    parser.add_argument("--num-classes", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out-dir", default="./runs/moment_large_shl2023_ft")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    main(args)
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MOMENT-1-large を SHL-2023 で fine-tuning する最小実行スクリプト
（後で教師モデルとして使えるように、学習済み重みを保存）

依存:
    pip install torch torchvision timm momentfm scikit-learn scipy numpy
"""

import os, math, time, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from scipy import signal

# MOMENT
from momentfm.models.moment import MOMENT


# --------------------------
# データ前処理と Dataset
# --------------------------
class CustomTransform:
    def __init__(self, signal_length, is_training=True, jitter_strength=0.1, flip_prob=0.5):
        self.signal_length = signal_length
        self.is_training = is_training
        self.jitter_strength = jitter_strength
        self.flip_prob = flip_prob

    def __call__(self, x):
        y = np.zeros((x.shape[0], self.signal_length), dtype=np.float32)
        for i in range(x.shape[0]):
            y[i] = signal.resample(x[i], self.signal_length)
        y = torch.from_numpy(y).float()
        if self.is_training:
            y = y + torch.randn_like(y) * self.jitter_strength
            if random.random() < self.flip_prob:
                y = torch.flip(y, dims=[-1])
        return y


class ResampleToTensor:
    def __init__(self, signal_length):
        self.signal_length = signal_length
    def __call__(self, x):
        y = np.zeros((x.shape[0], self.signal_length), dtype=np.float32)
        for i in range(x.shape[0]):
            y[i] = signal.resample(x[i], self.signal_length)
        return torch.from_numpy(y).float()


class TriaxialSignalDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        x, y = self.data[idx], self.labels[idx]
        if self.transform: x = self.transform(x)
        return x, y


# --------------------------
# モデル定義
# --------------------------
class MOMENTForClassification(nn.Module):
    """MOMENT encoder + Linear head"""
    def __init__(self, moment_id, num_classes, input_size):
        super().__init__()
        self.backbone = MOMENT.from_pretrained(moment_id)
        hidden = getattr(self.backbone.config, "hidden_size", 512)
        self.head = nn.Linear(hidden, num_classes)

    def _forward_features(self, x):
        out = self.backbone(x)
        if isinstance(out, dict):
            for key in ["cls", "pooled", "features", "last_hidden_state"]:
                if key in out:
                    feat = out[key]; break
            else:
                feat = next(iter(out.values()))
        else:
            feat = out
        if feat.dim() == 3:
            feat = feat.mean(dim=1)
        return feat

    def forward(self, x):
        feat = self._forward_features(x)
        return self.head(feat)


# --------------------------
# 学習関数
# --------------------------
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    def forward(self, preds, target):
        log_probs = F.log_softmax(preds, dim=-1)
        nll = F.nll_loss(log_probs, target, reduction='none')
        smooth = -log_probs.mean(dim=-1)
        return ((1 - self.smoothing) * nll + self.smoothing * smooth).mean()


@torch.no_grad()
def accuracy(output, target):
    pred = output.argmax(dim=1)
    correct = pred.eq(target).sum().item()
    return 100.0 * correct / len(target)


def cosine_lr(optimizer, base_lr, warmup, total_epochs, epoch):
    if epoch < warmup:
        lr = base_lr * (epoch + 1) / warmup
    else:
        lr = 0.5 * base_lr * (1 + math.cos(math.pi * (epoch - warmup) / (total_epochs - warmup)))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


# --------------------------
# メイン
# --------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(0); np.random.seed(0); torch.manual_seed(0)

    # データ読み込み
    acc = np.load(args.acc_path)
    lab = np.load(args.lab_path)
    lab = np.array([l - 1 for l in lab])
    tr_idx, va_idx = train_test_split(np.arange(len(lab)), train_size=0.9, random_state=0)
    train_tf = CustomTransform(signal_length=args.input_size, is_training=True)
    val_tf = ResampleToTensor(signal_length=args.input_size)
    ds_tr = TriaxialSignalDataset(acc[tr_idx], lab[tr_idx], transform=train_tf)
    ds_va = TriaxialSignalDataset(acc[va_idx], lab[va_idx], transform=val_tf)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=4)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # モデル
    model = MOMENTForClassification(args.moment_id, args.num_classes, args.input_size).to(device)
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)

    best_acc = 0
    for ep in range(args.epochs):
        model.train(); total_loss = 0
        lr_now = cosine_lr(optimizer, args.lr, 5, args.epochs, ep)
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward(); optimizer.step()
            total_loss += loss.item() * x.size(0)
        train_loss = total_loss / len(ds_tr)

        # 評価
        model.eval(); total_acc = 0; total_loss = 0
        with torch.no_grad():
            for x, y in dl_va:
                x, y = x.to(device), y.to(device)
                out = model(x)
                total_loss += criterion(out, y).item() * x.size(0)
                total_acc += accuracy(out, y) * x.size(0)
        val_loss = total_loss / len(ds_va)
        val_acc = total_acc / len(ds_va)

        print(f"Epoch {ep+1}/{args.epochs}  lr={lr_now:.6f}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best_ckpt.pth"))
            print(f"  [saved best_ckpt.pth] acc={best_acc:.2f}%")

    print("done.")


# --------------------------
# CLI
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--acc-path", required=True, help="Hips_Acc.npy のパス")
    parser.add_argument("--lab-path", required=True, help="Hips_Label.npy のパス")
    parser.add_argument("--moment-id", default="AutonLab/MOMENT-1-large")
    parser.add_argument("--input-size", type=int, default=496)
    parser.add_argument("--num-classes", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out-dir", default="./runs/moment_large_shl2023_ft")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    main(args)
