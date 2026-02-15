import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from loss_func import DistillationLoss
import utils

def _all_finite(t: torch.Tensor) -> bool:
    return torch.isfinite(t).all().item()

def train_one_epoch(
    model: torch.nn.Module,
    criterion: DistillationLoss,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    model_ema: Optional[ModelEma] = None,
    mixup_fn: Optional[Mixup] = None,
    set_training_mode=True,
    args=None,
):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    if args.cosub:
        criterion = torch.nn.BCEWithLogitsLoss()

    step_idx = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if not _all_finite(samples):
            print(f"[warn] non-finite samples at step {step_idx}, skip this batch")
            step_idx += 1
            continue
        if not _all_finite(targets.float()):
            print(f"[warn] non-finite targets at step {step_idx}, skip this batch")
            step_idx += 1
            continue

        if args.student == "DeepConvLSTM":
            samples = samples.permute(0, 2, 1)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if args.cosub:
            samples = torch.cat((samples, samples), dim=0)

        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)

        with torch.amp.autocast("cuda"):
            outputs = model(samples)

            if isinstance(outputs, (tuple, list)):
                stu_logits = outputs[0]
            else:
                stu_logits = outputs
            if not _all_finite(stu_logits):
                print(f"[warn] non-finite student logits at step {step_idx}, skip this batch")
                step_idx += 1
                continue

            if not args.cosub:
                loss = criterion(samples, outputs, targets)
            else:
                outputs = torch.split(outputs, outputs.shape[0] // 2, dim=0)
                loss = 0.25 * criterion(outputs[0], targets)
                loss = loss + 0.25 * criterion(outputs[1], targets)
                loss = loss + 0.25 * criterion(outputs[0], outputs[1].detach().sigmoid())
                loss = loss + 0.25 * criterion(outputs[1], outputs[0].detach().sigmoid())

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"[warn] loss is {loss_value} at step {step_idx}, skip this batch")
            optimizer.zero_grad(set_to_none=True)
            step_idx += 1
            continue

        optimizer.zero_grad(set_to_none=True)
        is_second_order = hasattr(optimizer, "is_second_order") and optimizer.is_second_order
        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=is_second_order,
        )

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if hasattr(criterion, "last") and criterion.last and step_idx % print_freq == 0:
            kd = criterion.last
            if kd["kd_loss"] is not None:
                print(
                    "KD base={:.4f} kd={:.4f} tH={:.3f} sH={:.3f} agree={:.3f} alpha={:.3f} tau={:.3f}".format(
                        kd["base_loss"],
                        kd["kd_loss"],
                        kd["t_entropy"],
                        kd["s_entropy"],
                        kd["agree"],
                        kd["alpha"],
                        kd["tau"],
                    )
                )
        step_idx += 1

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            output = model(images)
            if isinstance(output, (tuple, list)):
                output = output[0]  # logits
            loss = criterion(output, target)
        

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)

    metric_logger.synchronize_between_processes()
    print(
        "* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}".format(
            top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss
        )
    )
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
