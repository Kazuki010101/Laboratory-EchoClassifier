import math
import time
from typing import Iterable, Optional

import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from loss_func import DistillationLoss
import utils

def _unwrap_model(model):
    if hasattr(model, "module"):
        return model.module

    return model


def _get_model_aux_loss(model):
    core_model = _unwrap_model(model)

    if not hasattr(core_model, "get_aux_loss"):
        return None

    return core_model.get_aux_loss()


def _get_model_routing_stats(model):
    core_model = _unwrap_model(model)

    if not hasattr(core_model, "get_routing_stats"):
        return {}

    return core_model.get_routing_stats()

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
    epoch_start = time.time()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model.train(set_training_mode)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "lr",
        utils.SmoothedValue(
            window_size=1,
            fmt="{value:.6f}",
        ),
    )

    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    if args.cosub:
        criterion = torch.nn.BCEWithLogitsLoss()

    step_idx = 0

    for samples, targets in metric_logger.log_every(
        data_loader,
        print_freq,
        header,
    ):
        samples = samples.to(
            device,
            non_blocking=True,
        )
        targets = targets.to(
            device,
            non_blocking=True,
        )

        if not _all_finite(samples):
            print(
                f"[warn] non-finite samples at step "
                f"{step_idx}, skip this batch"
            )
            step_idx += 1
            continue

        if not _all_finite(targets.float()):
            print(
                f"[warn] non-finite targets at step "
                f"{step_idx}, skip this batch"
            )
            step_idx += 1
            continue

        if args.student == "DeepConvLSTM":
            samples = samples.permute(0, 2, 1)

        if mixup_fn is not None:
            samples, targets = mixup_fn(
                samples,
                targets,
            )

        if args.cosub:
            samples = torch.cat(
                (samples, samples),
                dim=0,
            )

        if args.bce_loss:
            targets = targets.gt(0.0).type(
                targets.dtype
            )

        routing_aux_loss = None

        with torch.amp.autocast(
            "cuda",
            enabled=torch.cuda.is_available(),
        ):
            outputs = model(samples)

            if isinstance(outputs, (tuple, list)):
                stu_logits = outputs[0]
            else:
                stu_logits = outputs

            if not _all_finite(stu_logits):
                print(
                    f"[warn] non-finite student logits at "
                    f"step {step_idx}, skip this batch"
                )
                step_idx += 1
                continue

            if not args.cosub:
                loss = criterion(
                    samples,
                    outputs,
                    targets,
                )
            else:
                split_outputs = torch.split(
                    outputs,
                    outputs.shape[0] // 2,
                    dim=0,
                )

                loss = 0.25 * criterion(
                    split_outputs[0],
                    targets,
                )
                loss = loss + 0.25 * criterion(
                    split_outputs[1],
                    targets,
                )
                loss = loss + 0.25 * criterion(
                    split_outputs[0],
                    split_outputs[1]
                    .detach()
                    .sigmoid(),
                )
                loss = loss + 0.25 * criterion(
                    split_outputs[1],
                    split_outputs[0]
                    .detach()
                    .sigmoid(),
                )

            routing_aux_loss = _get_model_aux_loss(
                model
            )

            if routing_aux_loss is not None:
                budget_weight = getattr(
                    args,
                    "innovation_budget_weight",
                    0.0,
                )

                loss = (
                    loss
                    + budget_weight
                    * routing_aux_loss
                )

        loss_value = loss.item()

        if step_idx % 100 == 0:
            with torch.no_grad():
                if isinstance(outputs, (tuple, list)):
                    debug_logits = outputs[0]
                else:
                    debug_logits = outputs

                pred = debug_logits.argmax(dim=1)

                if targets.ndim == 2:
                    debug_target = targets.argmax(
                        dim=1
                    )
                else:
                    debug_target = targets

                batch_acc = (
                    pred == debug_target
                ).float().mean().item()

                pred_unique, pred_counts = torch.unique(
                    pred.detach().cpu(),
                    return_counts=True,
                )

                target_unique, target_counts = torch.unique(
                    debug_target.detach().cpu(),
                    return_counts=True,
                )

                print(
                    f"[DEBUG][train]"
                    f"[epoch={epoch} step={step_idx}] "
                    f"loss={loss_value:.4f} "
                    f"batch_acc={batch_acc:.4f} "
                    f"logits_mean="
                    f"{debug_logits.mean().item():.4f} "
                    f"logits_std="
                    f"{debug_logits.std().item():.4f} "
                    f"target_min="
                    f"{debug_target.min().item()} "
                    f"target_max="
                    f"{debug_target.max().item()} "
                    f"pred_unique="
                    f"{list(zip(pred_unique.tolist(), pred_counts.tolist()))} "
                    f"target_unique="
                    f"{list(zip(target_unique.tolist(), target_counts.tolist()))}"
                )

                routing_stats = (
                    _get_model_routing_stats(model)
                )

                if routing_stats:
                    print(
                        "[DEBUG][routing] "
                        + " ".join(
                            f"{key}={value:.4f}"
                            for key, value
                            in routing_stats.items()
                        )
                    )

        if not math.isfinite(loss_value):
            print(
                f"[warn] loss is {loss_value} at "
                f"step {step_idx}, skip this batch"
            )
            optimizer.zero_grad(
                set_to_none=True
            )
            step_idx += 1
            continue

        optimizer.zero_grad(
            set_to_none=True
        )

        is_second_order = (
            hasattr(optimizer, "is_second_order")
            and optimizer.is_second_order
        )

        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=is_second_order,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(
            loss=loss_value
        )

        if routing_aux_loss is not None:
            metric_logger.update(
                innovation_budget_loss=float(
                    routing_aux_loss
                    .detach()
                    .item()
                )
            )

        routing_stats = _get_model_routing_stats(
            model
        )

        for key, value in routing_stats.items():
            metric_logger.update(
                **{
                    f"routing_{key}": float(value)
                }
            )

        metric_logger.update(
            lr=optimizer.param_groups[0]["lr"]
        )

        if (
            hasattr(criterion, "last")
            and criterion.last
            and step_idx % print_freq == 0
        ):
            kd = criterion.last

            if kd["kd_loss"] is not None:
                print(
                    "KD base={:.4f} "
                    "kd={:.4f} "
                    "tH={:.3f} "
                    "sH={:.3f} "
                    "agree={:.3f} "
                    "alpha={:.3f} "
                    "tau={:.3f}".format(
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

    print(
        "Averaged stats:",
        metric_logger,
    )

    epoch_time = (
        time.time() - epoch_start
    )

    stats = {
        key: meter.global_avg
        for key, meter
        in metric_logger.meters.items()
    }

    stats["epoch_time"] = epoch_time
    stats["num_steps"] = step_idx

    if torch.cuda.is_available():
        stats["max_memory_mb"] = (
            torch.cuda.max_memory_allocated()
            / 1024
            / 1024
        )
    else:
        stats["max_memory_mb"] = 0.0

    return stats


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    model.eval()
    debug_eval_done = False
    
    all_preds = []
    all_targets = []

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            output = model(images)

            if isinstance(output, (tuple, list)):
                output = output[0]

            loss = criterion(output, target)

        # ===== Evaluation Debug =====
        if not debug_eval_done:
            pred = output.argmax(dim=1)

            pred_unique, pred_counts = torch.unique(
                pred.detach().cpu(), return_counts=True
            )
            target_unique, target_counts = torch.unique(
                target.detach().cpu(), return_counts=True
            )

            print(
                "[DEBUG][eval] "
                f"output_mean={output.mean().item():.4f} "
                f"output_std={output.std().item():.4f} "
                f"pred_unique={list(zip(pred_unique.tolist(), pred_counts.tolist()))} "
                f"target_unique={list(zip(target_unique.tolist(), target_counts.tolist()))}"
            )

            debug_eval_done = True

        
        pred = output.argmax(dim=1)
        all_preds.append(pred.detach().cpu())
        all_targets.append(target.detach().cpu())
        
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]

        metric_logger.update(
            loss=loss.item()
        )
        
        routing_stats = _get_model_routing_stats(
            model
        )
        
        for key, value in routing_stats.items():
            metric_logger.update(
                **{
                    f"routing_{key}": float(value)
                }
            )
        
        metric_logger.meters["acc1"].update(
            acc1.item(),
            n=batch_size,
        )
        
        metric_logger.meters["acc5"].update(
            acc5.item(),
            n=batch_size,
        )

    metric_logger.synchronize_between_processes()

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    
    precision_macro = precision_score(
        all_targets, all_preds, average="macro", zero_division=0
    )
    recall_macro = recall_score(
        all_targets, all_preds, average="macro", zero_division=0
    )
    f1_macro = f1_score(
        all_targets, all_preds, average="macro", zero_division=0
    )
    
    print(
        "* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} "
        "loss {losses.global_avg:.3f} Precision {precision:.3f} Recall {recall:.3f} F1 {f1:.3f}".format(
            top1=metric_logger.acc1,
            top5=metric_logger.acc5,
            losses=metric_logger.loss,
            precision=precision_macro,
            recall=recall_macro,
            f1=f1_macro,
        )
    )
    
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    stats["precision_macro"] = float(precision_macro)
    stats["recall_macro"] = float(recall_macro)
    stats["f1_macro"] = float(f1_macro)
    
    return stats