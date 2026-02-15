import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import os
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 温度・損失関連関数
# ============================================================
def phi_temperature(T, gamma=1.0):
    return torch.exp(-gamma * T)

def compute_sigma_from_teacher_logits(zT):
    z_prev = torch.roll(zT, 1, dims=1)
    z_prev[:, 0, :] = zT[:, 0, :]
    return torch.norm(zT - z_prev, p=2, dim=-1)

def compute_Tt_from_sigma(sigma, T0=1.0, alpha=0.5):
    return T0 + alpha * sigma

def softmax_with_temp(logits, T, dim=-1):
    if T.dim() == 2:
        T = T.unsqueeze(-1)
    return F.softmax(logits / T.clamp(min=1e-6), dim=dim)

def kd_kl_per_timestep(zT, zS, Tt):
    pT = softmax_with_temp(zT, Tt)
    pS = softmax_with_temp(zS, Tt)
    kl = torch.sum(pT * (torch.log(pT + 1e-12) - torch.log(pS + 1e-12)), dim=-1)
    return (Tt ** 2 * kl).mean()

def feature_l2_loss(featT, featS):
    loss = 0.0
    count = 0
    for k in featT:
        if k in featS:
            loss += F.mse_loss(featS[k], featT[k])
            count += 1
    return loss / max(count, 1)

def compute_kappa_step(Tt, beta=5.0, tau=1.0):
    return torch.sigmoid(beta * (Tt - tau))

def expected_cost_per_timestep(kappa_t, Cfull=1.0, Cskip=0.1):
    return kappa_t * Cfull + (1 - kappa_t) * Cskip


# ============================================================
# 教師 forward
# ============================================================
def forward_moment(moment_model, x):
    with torch.no_grad():
        output = moment_model(x_enc=x)
        logits = output.logits
        zT = logits.unsqueeze(1)
    return zT, {}

def forward_senvt(senvt_model, x):
    with torch.no_grad():
        zT = senvt_model(x)
        if zT.dim() == 2:
            zT = zT.unsqueeze(1)
    return zT, {}


# ============================================================
# トレーニングステップ（単一モデル）
# ============================================================
def training_step(batch, teacher, student, optim, criterion_ce, config, distill_target):
    x = batch["x"].to(device)
    y = batch["y"].to(device)
    B = x.size(0)

    # --- forward ---
    if distill_target == "moment":
        zT, featT = forward_moment(teacher, x)
        zS, featS = forward_moment(student, x)
    else:
        zT, featT = forward_senvt(teacher, x)
        zS, featS = forward_senvt(student, x)

    if zS.dim() == 2:
        zS = zS.unsqueeze(1)

    # --- 温度 ---
    mode = config.get("mode", "E-DTD")
    if mode == "fixed-T":
        Tt = torch.ones(B, 1, device=device) * config.get("T_fixed", 2.0)
    else:
        sigma = compute_sigma_from_teacher_logits(zT)
        Tt = compute_Tt_from_sigma(sigma)

    # --- 損失 ---
    L_dtd = kd_kl_per_timestep(zT, zS, Tt)
    L_feat = feature_l2_loss(featT, featS)

    if config.get("lambda_E", 0) > 0:
        kappa_t = compute_kappa_step(Tt)
        L_energy = expected_cost_per_timestep(kappa_t).mean()
    else:
        L_energy = torch.tensor(0.0, device=device)

    logits = zS.mean(dim=1)
    L_ce = criterion_ce(logits, y)

    loss = (
        config.get("lambda_out", 1.0) * L_dtd
        + config.get("lambda_feat", 1.0) * L_feat
        + config.get("lambda_E", 0) * L_energy
        + L_ce
    )

    optim.zero_grad()
    loss.backward()
    optim.step()

    return {
        "loss": loss.item(),
        "L_dtd": L_dtd.item(),
        "L_feat": L_feat.item(),
        "L_energy": L_energy.item(),
        "L_ce": L_ce.item(),
        "Tt": Tt.detach().cpu().numpy(),
    }


# ============================================================
# 検証関数
# ============================================================
def evaluate(val_loader, student):
    student.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            zS, _ = student(x)
            logits = zS.mean(dim=1)
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / total


# ============================================================
# トレーニングループ（単一または二系列一貫性対応）
# ============================================================
def train_with_validation(train_loader, val_loader,
                          teacher_moment=None, teacher_senvt=None,
                          student_moment=None, student_senvt=None,
                          config=None, distill_target="moment",
                          save_dir="./results"):
    """
    一貫性蒸留モードでは:
      config["use_consistency"] = True
      teacher_moment, teacher_senvt, student_moment, student_senvt を渡す
    単一蒸留モードでは:
      teacher, student を1セット渡す
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "Tt_logs"), exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    epochs = config.get("epochs", 50)
    log_file = os.path.join(save_dir, "train_log.txt")

    use_consistency = config.get("use_consistency", False)
    lambda_C = config.get("lambda_C", 0.5)
    best_acc, best_epoch = -np.inf, -1

    if not use_consistency:
        # 単一モデルモード
        student = student_moment or student_senvt
        teacher = teacher_moment or teacher_senvt
        optim = torch.optim.Adam(student.parameters(), lr=config.get("lr", 1e-4))

        for ep in range(epochs):
            student.train()
            total_loss = 0
            total_L_dtd, total_L_feat, total_L_energy, total_L_ce = 0, 0, 0, 0
            all_Tt = []

            for batch in tqdm(train_loader, desc=f"[{distill_target}] Epoch {ep+1}/{epochs}"):
                stats = training_step(batch, teacher, student, optim, criterion, config, distill_target)
                total_loss += stats["loss"]
                total_L_dtd += stats["L_dtd"]
                total_L_feat += stats["L_feat"]
                total_L_energy += stats["L_energy"]
                total_L_ce += stats["L_ce"]
                all_Tt.append(stats["Tt"])

            n_batches = len(train_loader)
            val_acc = evaluate(val_loader, student)

            log_line = (
                f"Epoch {ep+1}: "
                f"total_loss={total_loss/n_batches:.4f}, "
                f"L_dtd={total_L_dtd/n_batches:.4f}, "
                f"L_feat={total_L_feat/n_batches:.4f}, "
                f"L_energy={total_L_energy/n_batches:.4f}, "
                f"L_ce={total_L_ce/n_batches:.4f}, "
                f"val_acc={val_acc:.4f}\n"
            )

            print(log_line.strip())
            with open(log_file, "a") as f:
                f.write(log_line)

            np.save(os.path.join(save_dir, "Tt_logs", f"Tt_epoch{ep+1}.npy"), np.concatenate(all_Tt, axis=0))

            if val_acc > best_acc:
                best_acc, best_epoch = val_acc, ep
                torch.save({
                    "student": deepcopy(student.state_dict()),
                    "val_acc": val_acc,
                    "epoch": ep,
                    "config": config
                }, os.path.join(save_dir, f"best_{distill_target}.pt"))
                print(f" new best {distill_target} model saved (epoch {ep+1}, acc={val_acc:.4f})")


    else:
        # 二系列一貫性モード
        optim_m = torch.optim.Adam(student_moment.parameters(), lr=config.get("lr", 1e-4))
        optim_s = torch.optim.Adam(student_senvt.parameters(), lr=config.get("lr", 1e-4))

        for ep in range(epochs):
            student_moment.train()
            student_senvt.train()
            total_loss = 0
            total_L_dtd_m, total_L_feat_m, total_L_energy_m, total_L_ce_m = 0, 0, 0, 0
            total_L_dtd_s, total_L_feat_s, total_L_energy_s, total_L_ce_s = 0, 0, 0, 0
            total_L_cons = 0
            all_Tt = []

            for batch in tqdm(train_loader, desc=f"[Consistency] Epoch {ep+1}/{epochs}"):
                stats_m = training_step(batch, teacher_moment, student_moment, optim_m, criterion, config, "moment")
                stats_s = training_step(batch, teacher_senvt, student_senvt, optim_s, criterion, config, "senvt")

                x = batch["x"].to(device)
                zM, _ = student_moment(x)
                zS, _ = student_senvt(x)
                if zM.dim() == 2: zM = zM.unsqueeze(1)
                if zS.dim() == 2: zS = zS.unsqueeze(1)

                Tt_avg = torch.tensor((stats_m["Tt"] + stats_s["Tt"]) / 2, device=device)
                w_t = phi_temperature(Tt_avg)
                pM = softmax_with_temp(zM, Tt_avg)
                pS = softmax_with_temp(zS, Tt_avg)
                L_cons = torch.mean(w_t * torch.sum(pM * (torch.log(pM + 1e-12) - torch.log(pS + 1e-12)), dim=-1))

                total_loss += stats_m["loss"] + stats_s["loss"] + lambda_C * L_cons.item()
                total_L_cons += L_cons.item()
                total_L_dtd_m += stats_m["L_dtd"]; total_L_feat_m += stats_m["L_feat"]; total_L_energy_m += stats_m["L_energy"]; total_L_ce_m += stats_m["L_ce"]
                total_L_dtd_s += stats_s["L_dtd"]; total_L_feat_s += stats_s["L_feat"]; total_L_energy_s += stats_s["L_energy"]; total_L_ce_s += stats_s["L_ce"]
                all_Tt.append(Tt_avg.cpu().numpy())

            n_batches = len(train_loader)
            val_acc_m = evaluate(val_loader, student_moment)
            val_acc_s = evaluate(val_loader, student_senvt)
            avg_acc = (val_acc_m + val_acc_s) / 2

            log_line = (
                f"Epoch {ep+1}: total_loss={total_loss/n_batches:.4f}, "
                f"L_cons={total_L_cons/n_batches:.4f}, "
                f"M(L_dtd={total_L_dtd_m/n_batches:.4f}, L_feat={total_L_feat_m/n_batches:.4f}, L_E={total_L_energy_m/n_batches:.4f}, L_ce={total_L_ce_m/n_batches:.4f}), "
                f"S(L_dtd={total_L_dtd_s/n_batches:.4f}, L_feat={total_L_feat_s/n_batches:.4f}, L_E={total_L_energy_s/n_batches:.4f}, L_ce={total_L_ce_s/n_batches:.4f}), "
                f"valM={val_acc_m:.4f}, valS={val_acc_s:.4f}, avg={avg_acc:.4f}\n"
            )

            print(log_line.strip())
            with open(log_file, "a") as f:
                f.write(log_line)


            np.save(os.path.join(save_dir, "Tt_logs", f"Tt_epoch{ep+1}.npy"), np.concatenate(all_Tt, axis=0))

            if avg_acc > best_acc:
                best_acc, best_epoch = avg_acc, ep
                torch.save({
                    "student_moment": deepcopy(student_moment.state_dict()),
                    "student_senvt": deepcopy(student_senvt.state_dict()),
                    "val_acc": avg_acc,
                    "epoch": ep,
                    "config": config
                }, os.path.join(save_dir, "best_consistency.pt"))
                print(f" new best consistency model saved (epoch {ep+1}, acc={avg_acc:.4f})")

    print(f"\nBest epoch={best_epoch+1}, val_acc={best_acc:.4f}")
    return best_acc