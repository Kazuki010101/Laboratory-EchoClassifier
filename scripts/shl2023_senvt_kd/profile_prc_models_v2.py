import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from models.PatchEchoClassifier import PatchReservoir as PRC
from models.TeacherGuidedEvidenceCondensation import PatchReservoir as RoutedPRC


def tensor_state_stats(model):
    state = model.state_dict()
    state_elements = sum(t.numel() for t in state.values())
    state_bytes = sum(t.numel() * t.element_size() for t in state.values())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_parameters = sum(p.numel() for p in model.parameters() if not p.requires_grad)
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


def profile_one(factory, x, device, warmup, repeats, reservoir_size, patch_size, is_tgec):
    model = factory().to(device).eval()
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

    stats = model.get_routing_stats() if hasattr(model, "get_routing_stats") else {
        "patch_count": x.shape[-1] // patch_size,
        "keep_count": x.shape[-1] // patch_size,
    }
    recurrent_tokens = int(stats["keep_count"]) + (1 if is_tgec else 0) + 2
    dominant_macs = recurrent_tokens * (
        reservoir_size**2 + patch_size * reservoir_size
    )
    result = {
        **tensor_state_stats(model),
        "latency_ms_median": statistics.median(times),
        "latency_ms_mean": statistics.mean(times),
        "latency_ms_stdev": statistics.stdev(times) if len(times) > 1 else 0.0,
        "peak_memory_mb": (
            torch.cuda.max_memory_allocated(device) / (2**20)
            if device.type == "cuda"
            else None
        ),
        "recurrent_tokens_including_cls_dist": recurrent_tokens,
        "estimated_dominant_macs": dominant_macs,
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
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    args = parser.parse_args()

    device = torch.device(args.device)
    x = torch.randn(1, 3, args.input_size, device=device)
    factories = {
        "prc_full": (
            lambda: PRC(3, args.patch_size, args.patch_size, args.reservoir_size, 8),
            False,
        ),
        "aps_prc": (
            lambda: RoutedPRC(
                3, args.patch_size, args.patch_size, args.reservoir_size, 8,
                args.keep_ratio, "aps"
            ),
            False,
        ),
        "tg_skip_prc": (
            lambda: RoutedPRC(
                3, args.patch_size, args.patch_size, args.reservoir_size, 8,
                args.keep_ratio, "teacher_skip"
            ),
            False,
        ),
        "tgec_prc": (
            lambda: RoutedPRC(
                3, args.patch_size, args.patch_size, args.reservoir_size, 8,
                args.keep_ratio, "tgec"
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
            "batch_size": 1,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "note": "Each model is profiled separately; model state includes parameters and persistent buffers.",
        }
    }
    for name, (factory, is_tgec) in factories.items():
        result[name] = profile_one(
            factory, x, device, args.warmup, args.repeats,
            args.reservoir_size, args.patch_size, is_tgec
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

