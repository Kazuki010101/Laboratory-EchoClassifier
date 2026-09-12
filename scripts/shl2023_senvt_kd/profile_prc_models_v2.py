import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile


# Allow this file to be executed directly from any working directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.PatchEchoClassifier import PatchReservoir as PRC
from models.TeacherGuidedEvidenceCondensation import PatchReservoir as RoutedPRC


def tensor_state_stats(model):
    state = model.state_dict()
    state_elements = sum(t.numel() for t in state.values())
    state_bytes = sum(t.numel() * t.element_size() for t in state.values())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_parameters = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )
    buffers = sum(b.numel() for b in model.buffers())

    return {
        "trainable_parameters": trainable,
        "frozen_parameters": frozen_parameters,
        "buffer_elements": buffers,
        "model_state_elements": state_elements,
        "model_state_size_mb": state_bytes / (2**20),
    }


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_profiled_flops(model, x):
    """Count FLOPs for ATen operators supported by torch.profiler.

    The returned value is for one forward pass of the supplied batch. PyTorch
    counts a multiply and an addition as two floating-point operations.
    """
    with torch.inference_mode():
        with profile(
            activities=[ProfilerActivity.CPU],
            record_shapes=True,
            with_flops=True,
        ) as prof:
            model(x)

    return int(
        sum(
            int(getattr(event, "flops", 0) or 0)
            for event in prof.key_averages()
        )
    )


def profile_one(
    factory,
    x,
    device,
    warmup,
    repeats,
    reservoir_size,
    patch_size,
    is_tgec,
):
    # Count operations on CPU to avoid depending on CUDA/CUPTI profiler
    # availability. FLOP count is architecture-dependent, not device-dependent.
    model = factory().cpu().eval()
    flop_input = x.detach().cpu()
    profiled_flops = measure_profiled_flops(model, flop_input)

    model = model.to(device).eval()

    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        synchronize(device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        times = []
        for _ in range(repeats):
            synchronize(device)
            start = time.perf_counter()
            model(x)
            synchronize(device)
            times.append((time.perf_counter() - start) * 1000)

    stats = (
        model.get_routing_stats()
        if hasattr(model, "get_routing_stats")
        else {
            "patch_count": x.shape[-1] // patch_size,
            "keep_count": x.shape[-1] // patch_size,
        }
    )

    # The reservoir processes the selected patch-derived tokens, one summary
    # token for TGEC, and the CLS/distillation tokens.
    recurrent_tokens = int(stats["keep_count"]) + (1 if is_tgec else 0) + 2

    # Per-sample dominant reservoir cost. The expression counts MACs for the
    # recurrent and input projections at each reservoir update.
    dominant_macs = recurrent_tokens * (
        reservoir_size**2 + patch_size * reservoir_size
    )

    latency_mean = statistics.mean(times)
    batch_size = int(x.shape[0])

    result = {
        **tensor_state_stats(model),
        "latency_ms_median": statistics.median(times),
        "latency_ms_mean": latency_mean,
        "latency_ms_stdev": (
            statistics.stdev(times) if len(times) > 1 else 0.0
        ),
        "throughput_samples_per_sec": batch_size * 1000.0 / latency_mean,
        "profiled_flops_per_batch": profiled_flops,
        "profiled_flops_per_sample": profiled_flops / batch_size,
        "profiled_mflops_per_sample": profiled_flops / batch_size / 1e6,
        "peak_memory_mb": (
            torch.cuda.max_memory_allocated(device) / (2**20)
            if device.type == "cuda"
            else None
        ),
        "recurrent_tokens_including_cls_dist": recurrent_tokens,
        "estimated_dominant_macs_per_sample": dominant_macs,
        "estimated_dominant_mmacs_per_sample": dominant_macs / 1e6,
        "estimated_dominant_mflops_per_sample": 2 * dominant_macs / 1e6,
        **stats,
    }

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--input-size", type=int, default=496)
    parser.add_argument("--reservoir-size", type=int, default=1000)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--keep-ratio", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.repeats < 2:
        raise ValueError("--repeats must be at least 2")
    if args.cpu_threads < 1:
        raise ValueError("--cpu-threads must be at least 1")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    if device.type == "cpu":
        torch.set_num_threads(args.cpu_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # PyTorch permits setting this only before inter-op work begins.
            pass

    x = torch.randn(
        args.batch_size,
        3,
        args.input_size,
        device=device,
    )

    factories = {
        "prc_full": (
            lambda: PRC(
                3,
                args.patch_size,
                args.patch_size,
                args.reservoir_size,
                8,
            ),
            False,
        ),
        "aps_prc": (
            lambda: RoutedPRC(
                3,
                args.patch_size,
                args.patch_size,
                args.reservoir_size,
                8,
                args.keep_ratio,
                "aps",
            ),
            False,
        ),
        "tg_skip_prc": (
            lambda: RoutedPRC(
                3,
                args.patch_size,
                args.patch_size,
                args.reservoir_size,
                8,
                args.keep_ratio,
                "teacher_skip",
            ),
            False,
        ),
        "tgec_prc": (
            lambda: RoutedPRC(
                3,
                args.patch_size,
                args.patch_size,
                args.reservoir_size,
                8,
                args.keep_ratio,
                "tgec",
            ),
            True,
        ),
    }

    result = {
        "metadata": {
            "device": str(device),
            "input_size": args.input_size,
            "patch_size": args.patch_size,
            "requested_keep_ratio": args.keep_ratio,
            "batch_size": args.batch_size,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "cpu_threads": (
                args.cpu_threads if device.type == "cpu" else None
            ),
            "flops_note": (
                "profiled_flops contains executed ATen operations for which "
                "torch.profiler provides FLOP formulas; one MAC is counted "
                "as two FLOPs."
            ),
            "state_note": (
                "Model state includes parameters and persistent buffers."
            ),
        }
    }

    for name, (factory, is_tgec) in factories.items():
        result[name] = profile_one(
            factory,
            x,
            device,
            args.warmup,
            args.repeats,
            args.reservoir_size,
            args.patch_size,
            is_tgec,
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
