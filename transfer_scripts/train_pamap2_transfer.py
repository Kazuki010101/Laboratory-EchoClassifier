import argparse
import datetime
import json
import time
import sys
from pathlib import Path
from typing import Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import NativeScaler, ModelEma, get_state_dict

from datasets import build_dataset
from engine import train_one_epoch, evaluate
from loss_func import DistillationLoss
from samplers import RASampler
from augment4sig import new_data_aug_generator, Mixup1D

from models import PatchEchoClassifier
from models import PatchEchosAttnClassifier
from models import DeepConvLSTM
from models import resnet4sig
from models import MLPMixer4sig_dist

try:
    from models import PatchLowRankGatedReservoir
except Exception:
    PatchLowRankGatedReservoir = None

try:
    from models import PatchAdaptiveLowRankGatedReservoir
except Exception:
    PatchAdaptiveLowRankGatedReservoir = None

try:
    from models import PatchStateInnovationLRGR
except Exception:
    PatchStateInnovationLRGR = None

try:
    from models import PatchInputFactorizedStateAttentiveLRGR
except Exception:
    PatchInputFactorizedStateAttentiveLRGR = None

import utils


def get_args_parser():
    parser = argparse.ArgumentParser("PAMAP2 transfer fine-tuning", add_help=False)

    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--input-size", default=496, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--resume", default="")
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--eval", action="store_true")

    parser.add_argument(
        "--data",
        default="PAMAP2",
        choices=[
            "SHL2023",
            "SHL2024",
            "ADL",
            "PAMAP",
            "PAMAP2",
            "REALWORLD",
            "WISDM",
            "CAPTURE",
            "SHL2023_test",
        ],
    )
    parser.add_argument(
        "--pamap2_root",
        default="dataset/pamap2_100hz_w496_s248_chest16g",
        type=str,
    )
    parser.add_argument("--pamap2_drop_subj", default=109, type=int)
    parser.add_argument("--fold_test_subj", default=108, type=int)
    parser.add_argument("--fold_val_subj", default=107, type=int)
    parser.add_argument("--val_ratio", default=0.1, type=float)

    parser.add_argument("--student", default="PRC", type=str)
    parser.add_argument("--model", default="none", type=str)
    parser.add_argument("--patch_size", default=128, type=int)
    parser.add_argument("--reservoir_size", default=1000, type=int)
    parser.add_argument("--reservoir_rank", default=64, type=int)
    parser.add_argument(
        "--patch_keep_ratio",
        default=0.5,
        type=float,
    )
    
    parser.add_argument(
        "--innovation-threshold",
        default=0.5,
        type=float,
    )
    
    parser.add_argument(
        "--innovation-target-ratio",
        default=0.5,
        type=float,
    )
    
    parser.add_argument(
        "--innovation-budget-weight",
        default=0.1,
        type=float,
    )
    
    parser.add_argument(
        "--innovation-hidden-dim",
        default=64,
        type=int,
    )
    
    parser.add_argument(
        "--innovation-min-keep",
        default=1,
        type=int,
    )
    
    parser.add_argument(
        "--input_rank",
        "--input-rank",
        dest="input_rank",
        default=16,
        type=int,
    )

    parser.add_argument("--transfer-checkpoint", default="", type=str)
    parser.add_argument("--skip-head", action="store_true", default=True)
    parser.add_argument("--load-strict", action="store_true", default=False)

    parser.add_argument("--opt", default="adamw", type=str)
    parser.add_argument("--opt-eps", default=1e-8, type=float)
    parser.add_argument("--opt-betas", default=None, type=float, nargs="+")
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.05)

    parser.add_argument("--sched", default="cosine", type=str)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--unscale-lr", action="store_true")
    parser.add_argument("--lr-noise", type=float, nargs="+", default=None)
    parser.add_argument("--lr-noise-pct", type=float, default=0.67)
    parser.add_argument("--lr-noise-std", type=float, default=1.0)
    parser.add_argument("--warmup-lr", type=float, default=1e-6)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--decay-epochs", type=float, default=30)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--cooldown-epochs", type=int, default=10)
    parser.add_argument("--patience-epochs", type=int, default=10)
    parser.add_argument("--decay-rate", "--dr", type=float, default=0.1)

    parser.add_argument("--smoothing", type=float, default=0.1)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--cutmix", type=float, default=0.0)
    parser.add_argument("--cutmix-minmax", type=float, nargs="+", default=None)
    parser.add_argument("--mixup-prob", type=float, default=1.0)
    parser.add_argument("--mixup-switch-prob", type=float, default=0.5)
    parser.add_argument("--mixup-mode", type=str, default="batch")
    parser.add_argument("--bce-loss", action="store_true")
    parser.add_argument("--ThreeAugment", action="store_true")
    parser.add_argument("--src", action="store_true")
    parser.add_argument("--flip_on", action="store_true", default=False)
    parser.add_argument("--color-jitter", type=float, default=0.3)
    parser.add_argument("--train-interpolation", type=str, default="bicubic")

    parser.add_argument(
        "--distillation-type",
        default="none",
        choices=["none", "soft", "hard", "soft2", "soft3", "soft4"],
        type=str,
    )
    parser.add_argument("--teacher-path", default="", type=str)
    parser.add_argument("--distillation-alpha", default=0.5, type=float)
    parser.add_argument("--distillation-tau", default=1.0, type=float)

    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--pin-mem", action="store_true")
    parser.add_argument("--no-pin-mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    parser.add_argument("--model-ema", action="store_true")
    parser.add_argument("--no-model-ema", action="store_false", dest="model_ema")
    parser.set_defaults(model_ema=False)
    parser.add_argument("--model-ema-decay", type=float, default=0.99996)
    parser.add_argument("--model-ema-force-cpu", action="store_true", default=False)

    parser.add_argument("--distributed", action="store_true", default=False)
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--dist-eval", action="store_true", default=False)

    parser.add_argument("--repeated-aug", action="store_true")
    parser.add_argument("--no-repeated-aug", action="store_false", dest="repeated_aug")
    parser.set_defaults(repeated_aug=True)

    parser.add_argument("--train-mode", action="store_true")
    parser.add_argument("--no-train-mode", action="store_false", dest="train_mode")
    parser.set_defaults(train_mode=True)

    parser.add_argument("--cosub", action="store_true")
    parser.add_argument("--data-path", default="", type=str)
    parser.add_argument("--inat-category", default="name")

    return parser


def ensure_compat_args(args):
    defaults = {
        "flip_on": False,
        "aa": None,
        "reprob": 0.0,
        "remode": "pixel",
        "recount": 1,
        "resplit": False,
        "hflip": 0.0,
        "vflip": 0.0,
    }

    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)

    return args


def build_deepconvlstm_config(student: str, input_size: int, num_classes: int) -> Dict:
    if "100" in student:
        return {
            "n_hidden": 128,
            "n_layers": 1,
            "n_filters": 64,
            "n_classes": num_classes,
            "filter_size": 5,
            "window_size": input_size,
            "channels": 3,
            "drop_prob": 0.5,
        }

    if "50" in student:
        return {
            "n_hidden": 64,
            "n_layers": 1,
            "n_filters": 32,
            "n_classes": num_classes,
            "filter_size": 5,
            "window_size": input_size,
            "channels": 3,
            "drop_prob": 0.5,
        }

    if "25" in student:
        return {
            "n_hidden": 32,
            "n_layers": 1,
            "n_filters": 16,
            "n_classes": num_classes,
            "filter_size": 5,
            "window_size": input_size,
            "channels": 3,
            "drop_prob": 0.5,
        }

    raise ValueError(f"Unknown DeepConvLSTM student: {student}")


def build_resnet(student: str, num_classes: int):
    if "L" in student:
        base_filters, n_block = 64, 8
    elif "M" in student:
        base_filters, n_block = 32, 8
    elif "S" in student:
        base_filters, n_block = 16, 4
    else:
        raise ValueError(f"Unknown Resnet student: {student}")

    return resnet4sig.ResNet1D(
        in_channels=3,
        base_filters=base_filters,
        kernel_size=7,
        stride=2,
        groups=1,
        n_block=n_block,
        n_classes=num_classes,
        downsample_gap=2,
        increasefilter_gap=4,
        use_bn=True,
        use_do=True,
        verbose=False,
    )


def build_student_model(args):
    if args.student == "MLPMixer":
        print("Creating student: MLPMixer")
        return MLPMixer4sig_dist.DistilledMLPMixer(
            dim=512,
            num_classes=args.nb_classes,
            depth=8,
        )

    if args.student == "PRC":
        print("Creating student: PRC")
        return PatchEchoClassifier.PatchReservoir(
            in_channels=3,
            patch_size=args.patch_size,
            stride=args.patch_size,
            reservoir_size=args.reservoir_size,
            num_classes=args.nb_classes,
        )

    if args.student == "PESAC":
        print("Creating student: PESAC")
        return PatchEchosAttnClassifier.PatchReservoir(
            in_channels=3,
            patch_size=args.patch_size,
            stride=args.patch_size,
            reservoir_size=args.reservoir_size,
            num_classes=args.nb_classes,
        )

    if args.student == "PRC_LRGR":
        if PatchLowRankGatedReservoir is None:
            raise ImportError("models/PatchLowRankGatedReservoir.py が見つかりません。")

        print("Creating student: PRC_LRGR")
        return PatchLowRankGatedReservoir.PatchReservoir(
            in_channels=3,
            patch_size=args.patch_size,
            stride=args.patch_size,
            reservoir_size=args.reservoir_size,
            reservoir_rank=args.reservoir_rank,
            num_classes=args.nb_classes,
        )

    if args.student == "APS_LRGR":
        if PatchAdaptiveLowRankGatedReservoir is None:
            raise ImportError("models/PatchAdaptiveLowRankGatedReservoir.py が見つかりません。")

        print("Creating student: APS_LRGR")
        return PatchAdaptiveLowRankGatedReservoir.PatchReservoir(
            in_channels=3,
            patch_size=args.patch_size,
            stride=args.patch_size,
            reservoir_size=args.reservoir_size,
            reservoir_rank=args.reservoir_rank,
            patch_keep_ratio=args.patch_keep_ratio,
            num_classes=args.nb_classes,
        )

    if args.student == "SIR_LRGR":
        if PatchStateInnovationLRGR is None:
            raise ImportError(
                "models/PatchStateInnovationLRGR.py が見つかりません。"
            )
    
        print("Creating student: SIR_LRGR")
    
        return PatchStateInnovationLRGR.PatchReservoir(
            in_channels=3,
            patch_size=args.patch_size,
            stride=args.patch_size,
            reservoir_size=args.reservoir_size,
            reservoir_rank=args.reservoir_rank,
            num_classes=args.nb_classes,
            router_hidden_dim=args.innovation_hidden_dim,
            routing_threshold=args.innovation_threshold,
            target_keep_ratio=args.innovation_target_ratio,
            minimum_keep_patches=args.innovation_min_keep,
        )
        
    if args.student == "IFSA_LRGR":
        if PatchInputFactorizedStateAttentiveLRGR is None:
            raise ImportError("models/PatchInputFactorizedStateAttentiveLRGR.py が見つかりません。")

        print("Creating student: IFSA_LRGR")
        return PatchInputFactorizedStateAttentiveLRGR.PatchReservoir(
            in_channels=3,
            patch_size=args.patch_size,
            stride=args.patch_size,
            reservoir_size=args.reservoir_size,
            reservoir_rank=args.reservoir_rank,
            input_rank=args.input_rank,
            num_classes=args.nb_classes,
        )

    if "DeepConvLSTM" in args.student:
        print(f"Creating student: {args.student}")
        config = build_deepconvlstm_config(
            student=args.student,
            input_size=args.input_size,
            num_classes=args.nb_classes,
        )
        return DeepConvLSTM.DeepConvLSTM(**config)

    if "Resnet" in args.student:
        print(f"Creating student: {args.student}")
        return build_resnet(args.student, args.nb_classes)

    raise ValueError(f"Unknown student model: {args.student}")


def normalize_checkpoint_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    new_state = {}

    for key, value in state_dict.items():
        k = key

        if k.startswith("module."):
            k = k[len("module."):]

        if k.startswith("core."):
            k = k[len("core."):]

        new_state[k] = value

    return new_state


def should_skip_transfer_key(key: str, args) -> bool:
    if not args.skip_head:
        return False

    head_keywords = [
        "classification_head",
        "distillation_head",
        "head",
        "fc",
        "classifier",
    ]

    return any(word in key for word in head_keywords)


def load_transfer_checkpoint(model: torch.nn.Module, ckpt_path: str, args) -> Tuple[int, int]:
    if not ckpt_path:
        print("[Transfer] no transfer checkpoint. Training from scratch.")
        return 0, 0

    ckpt_path = Path(ckpt_path)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"transfer checkpoint not found: {ckpt_path}")

    print(f"[Transfer] loading checkpoint: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        source_state = checkpoint["model"]
    else:
        source_state = checkpoint

    source_state = normalize_checkpoint_keys(source_state)
    target_state = model.state_dict()

    loaded_state = {}
    skipped = {}

    for key, value in source_state.items():
        if should_skip_transfer_key(key, args):
            skipped[key] = "head skipped"
            continue

        if key not in target_state:
            skipped[key] = "not in target"
            continue

        if target_state[key].shape != value.shape:
            skipped[key] = f"shape mismatch: source={tuple(value.shape)}, target={tuple(target_state[key].shape)}"
            continue

        loaded_state[key] = value

    missing, unexpected = model.load_state_dict(loaded_state, strict=False)

    print(f"[Transfer] loaded keys: {len(loaded_state)}")
    print(f"[Transfer] skipped keys: {len(skipped)}")
    print(f"[Transfer] missing keys after load: {len(missing)}")
    print(f"[Transfer] unexpected keys after load: {len(unexpected)}")

    if args.load_strict and (missing or unexpected or skipped):
        raise RuntimeError(
            "Strict transfer loading failed. "
            f"missing={len(missing)}, unexpected={len(unexpected)}, skipped={len(skipped)}"
        )

    return len(loaded_state), len(skipped)


def save_checkpoint(
    path: Path,
    model_without_ddp,
    optimizer,
    lr_scheduler,
    epoch,
    loss_scaler,
    args,
    model_ema=None,
    extra=None,
):
    state = {
        "model": model_without_ddp.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "epoch": epoch,
        "scaler": loss_scaler.state_dict(),
        "args": args,
    }

    if model_ema is not None:
        state["model_ema"] = get_state_dict(model_ema)

    if extra:
        state.update(extra)

    utils.save_on_master(state, path)


def to_float_dict(stats: Dict) -> Dict:
    out = {}

    for key, value in stats.items():
        try:
            out[key] = float(value)
        except Exception:
            out[key] = value

    return out


def main(args):
    utils.init_distributed_mode(args)

    print(args)

    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    ret = build_dataset(args=args)

    if args.data == "PAMAP2":
        dataset_train, dataset_val, dataset_test, args.nb_classes = ret
    else:
        dataset_train, dataset_val, args.nb_classes = ret
        dataset_test = None

    print("===== Dataset =====")
    print("data:", args.data)
    print("num_train:", len(dataset_train))
    print("num_val:", len(dataset_val))
    print("num_test:", len(dataset_test) if dataset_test is not None else None)
    print("num_classes:", args.nb_classes)
    print("test_subject:", args.fold_test_subj if args.data == "PAMAP2" else None)
    print("val_subject:", args.fold_val_subj if args.data == "PAMAP2" else None)
    print("===================")

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()

        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train,
                num_replicas=num_tasks,
                rank=global_rank,
                shuffle=True,
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train,
                num_replicas=num_tasks,
                rank=global_rank,
                shuffle=True,
            )

        if args.dist_eval:
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val,
                num_replicas=num_tasks,
                rank=global_rank,
                shuffle=False,
            )
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    if args.ThreeAugment:
        data_loader_train.dataset.transform = new_data_aug_generator(args)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    data_loader_test = None

    if dataset_test is not None:
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test,
            sampler=torch.utils.data.SequentialSampler(dataset_test),
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )

    mixup_fn = None
    mixup_active = args.mixup > 0.0 or args.cutmix > 0.0 or args.cutmix_minmax is not None

    if mixup_active:
        mixup_fn = Mixup1D(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.nb_classes,
        )

    model = build_student_model(args)

    loaded_transfer_keys, skipped_transfer_keys = load_transfer_checkpoint(
        model=model,
        ckpt_path=args.transfer_checkpoint,
        args=args,
    )

    model.to(device)

    teacher_model = None

    if args.distillation_type != "none":
        raise NotImplementedError(
            "PAMAP2 transferではまず --distillation-type none を使ってください。"
            "SHL teacherとPAMAP2はクラスが違う可能性が高いためです。"
        )

    print("Distillation OFF")

    model_ema = None

    if args.model_ema:
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device="cpu" if args.model_ema_force_cpu else "",
            resume="",
        )

    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_parameters_total = sum(p.numel() for p in model.parameters())

    print("trainable params:", n_parameters)
    print("total params:", n_parameters_total)

    if not args.unscale_lr:
        args.lr = args.lr * args.batch_size * utils.get_world_size() / 512.0

    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()
    lr_scheduler, _ = create_scheduler(args, optimizer)

    if mixup_active:
        base_criterion = SoftTargetCrossEntropy()
    elif args.smoothing and args.smoothing > 0.0:
        base_criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        base_criterion = torch.nn.CrossEntropyLoss()

    if args.bce_loss:
        base_criterion = torch.nn.BCEWithLogitsLoss()

    criterion = DistillationLoss(
        base_criterion,
        teacher_model,
        args.distillation_type,
        args.distillation_alpha,
        args.distillation_tau,
    )

    output_dir = Path(args.output_dir)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model_without_ddp.load_state_dict(checkpoint["model"])

        if not args.eval:
            if "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if "lr_scheduler" in checkpoint:
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            if "epoch" in checkpoint:
                args.start_epoch = checkpoint["epoch"] + 1
            if "scaler" in checkpoint:
                loss_scaler.load_state_dict(checkpoint["scaler"])

        lr_scheduler.step(args.start_epoch)

    if args.eval:
        val_stats = evaluate(data_loader_val, model, device)
        print(f"[VAL] acc1={val_stats['acc1']:.2f} loss={val_stats['loss']:.4f}")

        if data_loader_test is not None:
            test_stats = evaluate(data_loader_test, model, device)
            print(f"[TEST] acc1={test_stats['acc1']:.2f} loss={test_stats['loss']:.4f}")

        return

    print(f"Start PAMAP2 transfer fine-tuning for {args.epochs} epochs")

    start_time = time.time()

    best_val_loss = float("inf")
    best_val_acc1 = 0.0
    best_epoch = -1

    elapsed_training_time = 0.0
    max_memory_mb_all = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.clip_grad,
            model_ema,
            mixup_fn,
            set_training_mode=args.train_mode,
            args=args,
        )

        elapsed_training_time += train_stats.get("epoch_time", 0.0)
        max_memory_mb_all = max(
            max_memory_mb_all,
            train_stats.get("max_memory_mb", 0.0),
        )

        lr_scheduler.step(epoch)

        if args.output_dir:
            save_checkpoint(
                output_dir / "checkpoint.pth",
                model_without_ddp,
                optimizer,
                lr_scheduler,
                epoch,
                loss_scaler,
                args,
                model_ema=model_ema,
            )

        val_stats = evaluate(data_loader_val, model, device)
        val_loss = float(val_stats["loss"])
        val_acc1 = float(val_stats["acc1"])

        print(f"[VAL] epoch={epoch} loss={val_loss:.4f} acc1={val_acc1:.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc1 = val_acc1
            best_epoch = epoch

            if args.output_dir:
                save_checkpoint(
                    output_dir / "best_checkpoint.pth",
                    model_without_ddp,
                    optimizer,
                    lr_scheduler,
                    epoch,
                    loss_scaler,
                    args,
                    model_ema=model_ema,
                    extra={
                        "best_val_loss": best_val_loss,
                        "best_val_acc1": best_val_acc1,
                    },
                )

        print(
            f"[BEST] epoch={best_epoch} "
            f"best_val_loss={best_val_loss:.4f} "
            f"best_val_acc1={best_val_acc1:.2f}"
        )

        kd_stats = {}

        if hasattr(criterion, "last") and criterion.last:
            kd_stats = {
                f"kd_{key}": value
                for key, value in criterion.last.items()
                if value is not None
            }

        log_stats = {
            **{f"train_{key}": value for key, value in train_stats.items()},
            **{f"val_{key}": value for key, value in val_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
            "n_parameters_total": n_parameters_total,
            "best_val_loss": best_val_loss,
            "best_val_acc1": best_val_acc1,
            "best_epoch": best_epoch,
            "elapsed_training_time": elapsed_training_time,
            "max_memory_mb_all": max_memory_mb_all,
            "transfer_checkpoint": args.transfer_checkpoint,
            "loaded_transfer_keys": loaded_transfer_keys,
            "skipped_transfer_keys": skipped_transfer_keys,
            "input_rank": args.input_rank,
            **kd_stats,
        }

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as file:
                file.write(json.dumps(to_float_dict(log_stats)) + "\n")

    test_stats = None

    if data_loader_test is not None and args.output_dir:
        best_path = output_dir / "best_checkpoint.pth"

        if best_path.exists():
            checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
            model_without_ddp.load_state_dict(checkpoint["model"])
            model.to(device)
            model.eval()

            test_stats = evaluate(data_loader_test, model, device)

            print(
                f"[TEST@BEST] "
                f"test_subj={args.fold_test_subj} "
                f"loss={test_stats['loss']:.4f} "
                f"acc1={test_stats['acc1']:.2f}"
            )

            if utils.is_main_process():
                with (output_dir / "test_best.json").open("w") as file:
                    json.dump(
                        {
                            "dataset": args.data,
                            "student": args.student,
                            "patch_size": args.patch_size,
                            "reservoir_size": args.reservoir_size,
                            "reservoir_rank": args.reservoir_rank,
                            "patch_keep_ratio": args.patch_keep_ratio,
                            "input_rank": args.input_rank,
                            "transfer_checkpoint": args.transfer_checkpoint,
                            "loaded_transfer_keys": loaded_transfer_keys,
                            "skipped_transfer_keys": skipped_transfer_keys,
                            "test_subject": int(args.fold_test_subj),
                            "val_subject": int(args.fold_val_subj),
                            "best_epoch": int(checkpoint.get("epoch", -1)),
                            "best_val_loss": float(checkpoint.get("best_val_loss", float("nan"))),
                            "best_val_acc1": float(checkpoint.get("best_val_acc1", float("nan"))),
                            **{f"test_{key}": float(value) for key, value in test_stats.items()},
                        },
                        file,
                        indent=2,
                    )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    print(f"Training time {total_time_str}")

    if args.output_dir and utils.is_main_process():
        summary = {
            "dataset": args.data,
            "student": args.student,
            "patch_size": args.patch_size,
            "reservoir_size": args.reservoir_size,
            "reservoir_rank": args.reservoir_rank,
            "patch_keep_ratio": args.patch_keep_ratio,
            "input_rank": args.input_rank,
            "transfer_checkpoint": args.transfer_checkpoint,
            "loaded_transfer_keys": loaded_transfer_keys,
            "skipped_transfer_keys": skipped_transfer_keys,
            "test_subject": args.fold_test_subj if args.data == "PAMAP2" else None,
            "val_subject": args.fold_val_subj if args.data == "PAMAP2" else None,
            "nb_classes": args.nb_classes,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "n_parameters": n_parameters,
            "n_parameters_total": n_parameters_total,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_val_acc1": best_val_acc1,
            "elapsed_training_time": elapsed_training_time,
            "total_wall_time": total_time,
            "max_memory_mb_all": max_memory_mb_all,
        }

        if test_stats is not None:
            summary.update({f"test_{key}": float(value) for key, value in test_stats.items()})

        with (output_dir / "summary.json").open("w") as file:
            json.dump(summary, file, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "PAMAP2 transfer fine-tuning",
        parents=[get_args_parser()],
    )

    args = parser.parse_args()
    args = ensure_compat_args(args)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)