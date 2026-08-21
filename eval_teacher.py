import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, SequentialSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from datasets import build_dataset
from models.senvt_adapters import SenvtTeacherAdapter


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", default="SHL2023", type=str)
    parser.add_argument("--input-size", default=496, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--seed", default=0, type=int)

    parser.add_argument("--model", default="senvt-B", type=str)
    parser.add_argument("--teacher-path", required=True, type=str)

    # datasets.py が参照する可能性のある引数
    parser.add_argument("--student", default="teacher_eval", type=str)
    parser.add_argument("--flip_on", action="store_true")
    parser.add_argument("--pin-mem", action="store_true")

    # PAMAP2用のダミー。SHL2023では使わない
    parser.add_argument("--pamap2-root", default="", type=str)
    parser.add_argument("--pamap2-drop-subj", default=None, type=int)
    parser.add_argument("--fold-test-subj", default=101, type=int)
    parser.add_argument("--val-ratio", default=0.1, type=float)

    return parser.parse_args()


@torch.no_grad()
def main():
    args = get_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    dataset_train, dataset_val, nb_classes = build_dataset(args)
    print("num_train:", len(dataset_train))
    print("num_val:", len(dataset_val))
    print("num_classes:", nb_classes)

    data_loader_val = DataLoader(
        dataset_val,
        sampler=SequentialSampler(dataset_val),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    variant = args.model.replace("senvt-", "")
    teacher = SenvtTeacherAdapter(
        variant=variant,
        num_classes=nb_classes,
        window_size=args.input_size,
        in_chans=3,
        ckpt_path=args.teacher_path,
        device=device,
        verbose_ckpt=True,
    )
    teacher.eval()

    all_preds = []
    all_targets = []
    all_entropy = []

    debug_done = False

    for samples, targets in data_loader_val:
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # NaN/Infが混じっていたら確認用に置換
        samples = torch.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)

        logits = teacher(samples)

        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        probs = torch.softmax(logits, dim=1)
        entropy = (-probs * torch.log(torch.clamp(probs, min=1e-12))).sum(dim=1)

        preds = logits.argmax(dim=1)

        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())
        all_entropy.append(entropy.cpu())

        if not debug_done:
            pred_unique, pred_counts = torch.unique(preds.cpu(), return_counts=True)
            target_unique, target_counts = torch.unique(targets.cpu(), return_counts=True)

            print("[DEBUG][teacher]")
            print("logits_mean:", logits.mean().item())
            print("logits_std:", logits.std().item())
            print("entropy_mean:", entropy.mean().item())
            print("entropy_max_possible:", np.log(nb_classes))
            print("pred_unique:", list(zip(pred_unique.tolist(), pred_counts.tolist())))
            print("target_unique:", list(zip(target_unique.tolist(), target_counts.tolist())))

            debug_done = True

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    all_entropy = torch.cat(all_entropy).numpy()

    acc = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_targets, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    print("========== Teacher Eval ==========")
    print("teacher:", args.model)
    print("teacher_path:", args.teacher_path)
    print("acc:", acc)
    print("precision_macro:", precision)
    print("recall_macro:", recall)
    print("f1_macro:", f1)
    print("entropy_mean:", float(all_entropy.mean()))
    print("entropy_std:", float(all_entropy.std()))
    print("entropy_max_possible:", float(np.log(nb_classes)))


if __name__ == "__main__":
    main()
