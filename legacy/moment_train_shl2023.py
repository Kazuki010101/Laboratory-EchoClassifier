# filename: moment_train_shl2023.py
import argparse
import datetime
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import NativeScaler, get_state_dict, ModelEma

# 既存プロジェクトのユーティリティ群
import utils
from datasets import build_dataset
from engine import train_one_epoch, evaluate
from samplers import RASampler
from augment4sig import new_data_aug_generator, Mixup1D

# 重要：engine側の3引数契約に合わせるため、常にこのラッパーで包む
from loss_func import DistillationLoss

# MOMENT を “学習対象” として使うアダプタ
from moment_adapters import MomentStudentAdapter


def get_args():
    p = argparse.ArgumentParser("Train MOMENT on SHL-2023 (no distillation, drop-in compatible)")

    # ===== 基本 / データセット =====
    p.add_argument('--data', default='SHL2023',
                   choices=['SHL2023', 'SHL2024', 'ADL', 'PAMAP', 'REALWORLD', 'WISDM', 'CAPTURE', 'SHL2023_test'])
    p.add_argument('--input-size', type=int, default=496)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--num_workers', type=int, default=10)
    p.add_argument('--pin-mem', action='store_true')
    p.add_argument('--no-pin-mem', action='store_false', dest='pin_mem')
    p.set_defaults(pin_mem=True)
    p.add_argument('--output_dir', type=str, default='./runs/moment_large_shl2023_ft')

    # ===== MOMENT バリアント =====
    p.add_argument('--moment_id', type=str, default='AutonLab/MOMENT-1-large',
                   help='e.g., AutonLab/MOMENT-1-large or AutonLab/MOMENT-1-small')
    p.add_argument('--in_chans', type=int, default=3)

    # ===== Optim / Scheduler（timm 準拠）=====
    p.add_argument('--opt', default='adamw', type=str)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=0.05)
    p.add_argument('--sched', default='cosine', type=str)
    p.add_argument('--warmup-epochs', type=int, default=5)
    p.add_argument('--min-lr', type=float, default=1e-5)
    p.add_argument('--decay-epochs', type=float, default=30)
    p.add_argument('--decay-rate', type=float, default=0.1)
    p.add_argument('--unscale-lr', action='store_true')
    p.add_argument('--clip-grad', type=float, default=None)

    # timm が参照する互換引数
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--opt_eps', type=float, default=1e-8)
    p.add_argument('--opt-betas', type=float, nargs='+', default=None)

    # ===== Aug / Transform（datasets.build_transform が参照しがちな項目）=====
    p.add_argument('--smoothing', type=float, default=0.1)
    p.add_argument('--mixup', type=float, default=0.2)
    p.add_argument('--cutmix', type=float, default=0.0)
    p.add_argument('--mixup-prob', type=float, default=1.0)
    p.add_argument('--mixup-switch-prob', type=float, default=0.0)
    p.add_argument('--mixup-mode', type=str, default='batch')
    p.add_argument('--ThreeAugment', action='store_true')
    p.add_argument('--flip_on', action='store_true', default=False)
    p.add_argument('--train-interpolation', type=str, default='bicubic')
    p.add_argument('--color-jitter', type=float, default=0.3)
    p.add_argument('--src', action='store_true', default=False)

    # ===== 互換フラグ（他ファイルが参照しても落ちないように）=====
    p.add_argument('--student', type=str, default='moment-large')  # ダミー
    p.add_argument('--patch_size', type=int, default=16)
    p.add_argument('--reservoir_size', type=int, default=100)
    p.add_argument('--cosub', action='store_true', default=False)

    # ===== 蒸留系（今回未使用。参照されても落ちないために置く）=====
    p.add_argument('--distillation-type', type=str,
                   choices=['none', 'soft', 'hard', 'soft2', 'soft3', 'soft4'], default='none')
    p.add_argument('--distillation-alpha', type=float, default=0.5)
    p.add_argument('--distillation-tau', type=float, default=1.0)
    p.add_argument('--teacher-path', type=str, default='')
    p.add_argument('--bce-loss', action='store_true')

    # ===== EMA =====
    p.add_argument('--model-ema', action='store_true')
    p.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    p.set_defaults(model_ema=True)
    p.add_argument('--model-ema-decay', type=float, default=0.99996)
    p.add_argument('--model-ema-force-cpu', action='store_true', default=False)

    # ===== 分散（必要なら）=====
    p.add_argument('--distributed', action='store_true', default=False)
    p.add_argument('--world_size', type=int, default=1)
    p.add_argument('--dist_url', default='env://')

    # ===== 再開／評価 =====
    p.add_argument('--resume', default='')
    p.add_argument('--start_epoch', type=int, default=0)
    p.add_argument('--eval', action='store_true')

    # ===== 互換：repeated-aug / train-mode =====
    p.add_argument('--repeated-aug', action='store_true')
    p.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    p.set_defaults(repeated_aug=True)
    p.add_argument('--train-mode', action='store_true')
    p.add_argument('--no-train-mode', action='store_false', dest='train_mode')
    p.set_defaults(train_mode=True)

    return p.parse_args()


def _ema_state_dict(maybe_ema):
    try:
        return get_state_dict(maybe_ema) if maybe_ema is not None else None
    except Exception:
        return None


def main(args):
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # ===== Dataset =====
    dataset_train, dataset_val, args.nb_classes = build_dataset(args=args)

    # ===== Sampler =====
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        else:
            sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        sampler_val = torch.utils.data.DistributedSampler(dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # ===== DataLoader =====
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True
    )
    if args.ThreeAugment:
        data_loader_train.dataset.transform = new_data_aug_generator(args)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val, batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False
    )

    # ===== Mixup/Cutmix =====
    mixup_fn = None
    mixup_active = args.mixup > 0.0 or args.cutmix > 0.0
    if mixup_active:
        mixup_fn = Mixup1D(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=None,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes
        )

    # ===== Model（MOMENT を“学習対象”として使用）=====
    print(f"Creating MOMENT student: {args.moment_id}")
    model = MomentStudentAdapter(
        args.moment_id,
        n_channels=args.in_chans,
        num_class=args.nb_classes,
        device=device
    )
    model.to(device)

    # ===== EMA =====
    model_ema = None
    if args.model_ema:
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume=''
        )

    # ===== Optim/Scheduler =====
    model_without_ddp = model
    if args.distributed and hasattr(args, "gpu"):
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    if not args.unscale_lr:
        linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
        args.lr = linear_scaled_lr

    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()
    lr_scheduler, _ = create_scheduler(args, optimizer)

    # ===== Loss（ここが重要：常に DistillationLoss でラップ）=====
    base_criterion = (
        SoftTargetCrossEntropy() if mixup_active
        else (LabelSmoothingCrossEntropy(smoothing=args.smoothing) if args.smoothing and args.smoothing > 0
              else torch.nn.CrossEntropyLoss())
    )
    if args.bce_loss:
        base_criterion = torch.nn.BCEWithLogitsLoss()

    teacher_model = None  # 今回は蒸留しない
    criterion = DistillationLoss(
        base_criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    )

    # ===== Resume =====
    output_dir = Path(args.output_dir)
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(ckpt['model'])
        if not args.eval and 'optimizer' in ckpt and 'lr_scheduler' in ckpt and 'epoch' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
            lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
            args.start_epoch = ckpt['epoch'] + 1
            if args.model_ema and ckpt.get('model_ema', None) is not None:
                utils._load_checkpoint_for_ema(model_ema, ckpt['model_ema'])
            if 'scaler' in ckpt:
                loss_scaler.load_state_dict(ckpt['scaler'])
        lr_scheduler.step(args.start_epoch)

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy: {test_stats['acc1']:.2f}%")
        return

    # ===== Train Loop =====
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed and hasattr(data_loader_train, "sampler") and hasattr(data_loader_train.sampler, "set_epoch"):
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn,
            set_training_mode=args.train_mode, args=args
        )

        lr_scheduler.step(epoch)

        # save last
        if args.output_dir:
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'model_ema': (get_state_dict(model_ema) if model_ema is not None else None),
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }, output_dir / 'checkpoint.pth')

        # eval
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Val acc: {test_stats['acc1']:.2f}%")

        # save best
        if test_stats["acc1"] > max_accuracy:
            max_accuracy = test_stats["acc1"]
            if args.output_dir:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'model_ema': (get_state_dict(model_ema) if model_ema is not None else None),
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, output_dir / 'best_checkpoint.pth')

        # log
        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            'epoch': epoch,
            'n_parameters': n_parameters,
        }
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        print(f"Max accuracy so far: {max_accuracy:.2f}%")

    total_time = time.time() - start_time
    print('Training time {}'.format(str(datetime.timedelta(seconds=int(total_time)))))


if __name__ == "__main__":
    args = get_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
