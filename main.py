import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json

from pathlib import Path

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from datasets import build_dataset
from engine import train_one_epoch, evaluate
from loss_func import DistillationLoss
from samplers import RASampler
from augment4sig import new_data_aug_generator, Mixup1D

from models import MLPMixer4sig_dist
from models import transformer
from models import MLPMixer_for_signal
from models import PatchEchoClassifier
from models import DeepConvLSTM
from models import resnet4sig
# import PatchEchoAttnClassifier
from models import PatchEchosAttnClassifier

import utils

from models.moment_adapters import MomentTeacherAdapter, MomentStudentAdapter
from models.senvt_adapters import SenvtTeacherAdapter, SenvtStudentAdapter

def get_args_parser():
    parser = argparse.ArgumentParser('Training script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--bce-loss', action='store_true')
    parser.add_argument('--unscale-lr', action='store_true')

    # Model parameters
    
    parser.add_argument('--input-size', default=496, type=int, help='images input size')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    
    parser.add_argument('--model', default='Regnet', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--student', default='MLPMixer', type=str, metavar='MODEL',
                        help='Name of model')
    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--color-jitter', type=float, default=0.3, metavar='PCT',
                        help='Color jitter factor (default: 0.3)')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)
    
    parser.add_argument('--train-mode', action='store_true')
    parser.add_argument('--no-train-mode', action='store_false', dest='train_mode')
    parser.set_defaults(train_mode=True)
    
    parser.add_argument('--ThreeAugment', action='store_true') #3augment
    
    parser.add_argument('--src', action='store_true') #simple random crop

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard', 'soft2', 'soft3', 'soft4'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")
    
    # * Cosub params
    parser.add_argument('--cosub', action='store_true') 
    
    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--distributed', action='store_true', default=False, help='Enabling distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    # original
    parser.add_argument('--patch_size', default=16, type=int)
    parser.add_argument('--reservoir_size', default=100, type=int)
    parser.add_argument('--data', default='SHL', choices=['SHL2023', 'SHL2024', 'ADL', 'PAMAP', 'REALWORLD', 'WISDM', 'CAPTURE', 'SHL2023_test'])
    parser.add_argument('--flip_on', action='store_true', default=False)
    
    return parser


def main(args):
    utils.init_distributed_mode(args)

    print(args)

    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)


    cudnn.benchmark = True

    dataset_train, dataset_val, args.nb_classes = build_dataset(args=args)#変更

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    if args.ThreeAugment:
        data_loader_train.dataset.transform = new_data_aug_generator(args) #変更

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None #変更
    if mixup_active:
        mixup_fn = Mixup1D(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    if args.student == "MLPMixer":
        print(f"Creating model: MLPMixer")
        model = MLPMixer4sig_dist.DistilledMLPMixer(dim=512,num_classes=args.nb_classes, depth=8)#変更
    elif args.student == "PRC":
        print(f"Creating model: PatchReservoir")
        model = PatchEchoClassifier.PatchReservoir(in_channels=3, patch_size=args.patch_size, stride=args.patch_size, reservoir_size=args.reservoir_size, num_classes=args.nb_classes) #stride=1から変更
    elif args.student == "PEAC":
        print(f"Creating model: PatchAttnReservoir")
        model = PatchEchoAttnClassifier.PatchReservoir(in_channels=3, patch_size=args.patch_size, stride=args.patch_size, reservoir_size=args.reservoir_size, num_classes=args.nb_classes)
    elif args.student == "PESAC":
        print(f"Creating model: PatchSparseAttnReservoir")
        model = PatchEchosAttnClassifier.PatchReservoir(in_channels=3, patch_size=args.patch_size, stride=args.patch_size, reservoir_size=args.reservoir_size, num_classes=args.nb_classes)
    elif "DeepConvLSTM" in args.student:
        if "100" in args.student:
            config= {
            'n_hidden': 128,
            'n_layers': 1,  
            'n_filters': 64,
            'n_classes': 8,  
            'filter_size': 5,  
            'window_size': 496,  
            'channels': 3,  
            'drop_prob': 0.5,  
            }
        elif "50" in args.student:
            config = {
            'n_hidden': 64,  # 128 → 64
            'n_layers': 1,  
            'n_filters': 32,  # 64 → 32
            'n_classes': 8,  
            'filter_size': 5,  
            'window_size': 496,  
            'channels': 3,  
            'drop_prob': 0.5,  
            }
        elif "25" in args.student:
            config = {
            'n_hidden': 32,  # 128 → 32
            'n_layers': 1,  
            'n_filters': 16,  # 64 → 16
            'n_classes': 8,  
            'filter_size': 5,  
            'window_size': 496,  
            'channels': 3,  
            'drop_prob': 0.5,  
            }

        model = DeepConvLSTM.DeepConvLSTM(**config)
    elif args.student == "Transformer":
        print(f"Creating model: Transformer")
        model = transformer.L(num_classes=args.nb_classes)
    elif "Resnet" in args.student:
        if "L" in args.student:
            print(f"Creating model: Resnet_L")
            model = resnet4sig.ResNet1D(
                in_channels=3,
                base_filters=64,
                kernel_size=7,
                stride=2,
                groups=1,
                n_block=8,
                n_classes=args.nb_classes,
                downsample_gap=2,
                increasefilter_gap=4,
                use_bn=True,
                use_do=True,
                verbose=False
                )
        elif "M" in args.student:
            print(f"Creating model: Resnet_M")
            model = resnet4sig.ResNet1D(
                in_channels=3,
                base_filters=32,
                kernel_size=7,
                stride=2,
                groups=1,
                n_block=8,
                n_classes=args.nb_classes,
                downsample_gap=2,
                increasefilter_gap=4,
                use_bn=True,
                use_do=True,
                verbose=False
                )

        elif "S" in args.student:
            print(f"Creating model: Resnet_S")
            model = resnet4sig.ResNet1D(
                in_channels=3,
                base_filters=16,
                kernel_size=7,
                stride=2,
                groups=1,
                n_block=4,
                n_classes=args.nb_classes,
                downsample_gap=2,
                increasefilter_gap=4,
                use_bn=True,
                use_do=True,
                verbose=False
                )
    elif args.student == "moment-small":
        print("Creating student: MOMENT-small")
        model = MomentStudentAdapter(
            "AutonLab/MOMENT-1-small",
            n_channels=3, 
            num_class=args.nb_classes,
            device=device
        )

    elif args.student == "senvt-XS":
        print("Creating student: SENvT-XS")
        model = SenvtStudentAdapter(
            variant="XS",
            num_classes=args.nb_classes,
            window_size=args.input_size,   
            in_chans=3,
            device=device
        )
    elif args.student == "senvt-S":
        print("Creating student: SENvT-S")
        model = SenvtStudentAdapter(
            variant="S",
            num_classes=args.nb_classes,
            window_size=args.input_size,
            in_chans=3,
            device=device
        )

    else:
        raise ValueError(f"Unknown student model: {args.student}")

    model.to(device)

    class _KDWrapper(torch.nn.Module):
        def __init__(self, core):
            super().__init__()
            self.core = core
        def forward(self, x):
            out = self.core(x)
            if isinstance(out, tuple) and len(out) >= 1:
                return out
            if self.training:
                return out, out
            return out
    
    if args.distillation_type != 'none':
        if not isinstance(model, (MomentStudentAdapter, SenvtStudentAdapter)):
            model = _KDWrapper(model)

    teacher_model = None
    if args.distillation_type != 'none':
        print(f"Distillation ON: type={args.distillation_type}, alpha={args.distillation_alpha}, tau={args.distillation_tau}")

        if args.model == "Transformer":
            teacher_model = transformer.L(num_classes=args.nb_classes)
        elif args.model == "MLPMixer":
            teacher_model = MLPMixer_for_signal.MLPMixer(
                in_channels=3, 
                dim=768, 
                num_classes=args.nb_classes,
                patch_size=16, 
                sequence_length=496, 
                depth=12
            )
        elif args.model == "moment-large":  
            print("Creating teacher: MOMENT-1-large")
            teacher_model = MomentTeacherAdapter("AutonLab/MOMENT-1-large", n_channels=3, num_class=args.nb_classes, device=device)

        elif args.model == "senvt-B":
            print("Creating teacher: SENvT-B")
            teacher_model = SenvtTeacherAdapter(
                variant="B",
                num_classes=args.nb_classes,
                window_size=args.input_size,
                in_chans=3,
                ckpt_path="dataset/SENvT-u4/1000k_task4/best.pth",
                device=device
            )
        elif args.model == "senvt-L":
            print("Creating teacher: SENvT-L")
            teacher_model = SenvtTeacherAdapter(
                variant="L",
                num_classes=args.nb_classes,
                window_size=args.input_size,
                in_chans=3,
                ckpt_path="dataset/SENvT-u4/1000k_task4/best.pth",
                device=device
            )
        elif args.model == "senvt-XS":
            print("Creating teacher: SENvT-XS")
            teacher_model = SenvtTeacherAdapter(
                variant="XS",
                num_classes=args.nb_classes,
                window_size=args.input_size,
                in_chans=3,
                ckpt_path=args.teacher_path,   
                device=device
            )
        elif args.model == "senvt-S":
            print("Creating teacher: SENvT-S")
            teacher_model = SenvtTeacherAdapter(
                variant="S",
                num_classes=args.nb_classes,
                window_size=args.input_size,
                in_chans=3,
                ckpt_path=args.teacher_path,
                device=device
            )

        else:
            raise ValueError(f"Unknown teacher model: {args.model}")

        teacher_model.to(device)
        teacher_model.eval()
    else:
        print("Distillation OFF")
    
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    if args.distributed:
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

    criterion = LabelSmoothingCrossEntropy()

    if mixup_active:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
        
    if args.bce_loss:
        criterion = torch.nn.BCEWithLogitsLoss()

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'
    criterion = DistillationLoss(
        criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    )

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
        lr_scheduler.step(args.start_epoch)
    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn,
            set_training_mode=args.train_mode,
            args = args,
        )

        lr_scheduler.step(epoch)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'model_ema': get_state_dict(model_ema),
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)
             

        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        
        if max_accuracy < test_stats["acc1"]:
            max_accuracy = test_stats["acc1"]
            if args.output_dir:
                checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'model_ema': get_state_dict(model_ema),
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                    }, checkpoint_path)
            
        print(f'Max accuracy: {max_accuracy:.2f}%')

        kd_stats = {}
        if hasattr(criterion, "last") and criterion.last:
            kd_stats = {f'kd_{k}': v for k, v in criterion.last.items() if v is not None}

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters,
                     **kd_stats}
        
        
        
        
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
