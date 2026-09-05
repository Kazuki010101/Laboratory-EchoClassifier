"""Evaluate trained PAMAP2 checkpoints for accuracy, routing, latency and energy.

This script is intentionally separate from training.  It scans experiment
directories for ``best_checkpoint.pth``, reconstructs each model from the
arguments stored in the checkpoint, evaluates the held-out PAMAP2 subject, and
appends one reproducible row per checkpoint to a CSV file.

Energy backends:
  * NVIDIA GPU: NVML cumulative-energy counter when supported, otherwise a
    background power sampler (requires ``nvidia-ml-py`` / ``pynvml``).
  * Linux CPU: package-level RAPL counters under ``/sys/class/powercap``.

Run energy measurements only on an otherwise idle, dedicated device.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import gc
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from datasets import build_dataset
from transfer_scripts.train_pamap2_transfer import build_student_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained PAMAP2 models and create a paper-ready CSV."
    )
    parser.add_argument(
        "--experiments-root",
        action="append",
        required=True,
        help="Root containing best_checkpoint.pth files. May be supplied multiple times.",
    )
    parser.add_argument("--output-csv", required=True)
    parser.add_argument(
        "--summary-csv",
        default="",
        help=(
            "Subject-aggregated output CSV. By default, <output>_summary.csv "
            "is created next to --output-csv."
        ),
    )
    parser.add_argument(
        "--word-table-csv",
        default="",
        help="Optional compact baseline/transfer comparison CSV for the paper table.",
    )
    parser.add_argument(
        "--pamap2-root",
        default="dataset/pamap2_100hz_w496_s248_chest16g",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--model-filter",
        default="",
        help="Comma-separated model keys or glob patterns, e.g. APS_LRGR_*,SIR_LRGR_*.",
    )
    parser.add_argument("--checkpoint-filter", default="", help="Substring path filter.")
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--latency-samples", default=500, type=int)
    parser.add_argument("--latency-repeats", default=3, type=int)
    parser.add_argument("--warmup", default=50, type=int)
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--skip-memory", action="store_true")
    parser.add_argument("--skip-thop", action="store_true")
    parser.add_argument("--measure-energy", action="store_true")
    parser.add_argument("--energy-seconds", default=30.0, type=float)
    parser.add_argument("--idle-seconds", default=10.0, type=float)
    parser.add_argument("--power-sample-interval", default=0.02, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume-output",
        action="store_true",
        help=(
            "Keep existing rows in --output-csv, skip their checkpoint paths, "
            "and rebuild aggregate outputs from existing plus new rows."
        ),
    )
    return parser.parse_args()


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def args_to_dict(value) -> Dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    raise TypeError(f"Unsupported checkpoint args type: {type(value)!r}")


def default_eval_args() -> Dict:
    return {
        "data": "PAMAP2",
        "pamap2_root": "dataset/pamap2_100hz_w496_s248_chest16g",
        "pamap2_drop_subj": 109,
        "fold_test_subj": 108,
        "fold_val_subj": 107,
        "val_ratio": 0.1,
        "input_size": 496,
        "student": "PRC",
        "patch_size": 128,
        "reservoir_size": 1000,
        "reservoir_rank": 64,
        "patch_keep_ratio": 0.5,
        "input_rank": 16,
        "innovation_threshold": 0.5,
        "innovation_target_ratio": 0.5,
        "innovation_hidden_dim": 64,
        "innovation_min_keep": 1,
        "seed": 0,
        "flip_on": False,
        "ThreeAugment": False,
        "src": False,
        "color_jitter": 0.3,
        "train_interpolation": "bicubic",
    }


def model_key_from_checkpoint(path: Path) -> str:
    # Expected: <root>/<model_key>/testXXX_valYYY/best_checkpoint.pth
    return path.parent.parent.name


def experiment_type(path: Path) -> str:
    text = str(path).lower()
    if "transfer" in text:
        return "transfer"
    if "baseline" in text:
        return "baseline"
    return "unknown"


def find_checkpoints(args: argparse.Namespace) -> List[Path]:
    found: List[Path] = []
    wanted = [x.strip() for x in args.model_filter.split(",") if x.strip()]
    for root_text in args.experiments_root:
        root = Path(root_text)
        if not root.exists():
            print(f"[warn] experiments root does not exist: {root}")
            continue
        for path in root.rglob("best_checkpoint.pth"):
            key = model_key_from_checkpoint(path)
            if wanted and not any(fnmatch.fnmatchcase(key, pattern) for pattern in wanted):
                continue
            if args.checkpoint_filter and args.checkpoint_filter not in str(path):
                continue
            found.append(path)
    return sorted(set(found))


def make_eval_args(checkpoint: Dict, cli: argparse.Namespace) -> SimpleNamespace:
    values = default_eval_args()
    values.update(args_to_dict(checkpoint.get("args")))
    values["data"] = "PAMAP2"
    values["pamap2_root"] = cli.pamap2_root
    values["num_workers"] = cli.num_workers
    values["seed"] = values.get("seed", cli.seed)
    # Older checkpoints used underscore spellings; model builders use these names.
    aliases = {
        "innovation-threshold": "innovation_threshold",
        "innovation-target-ratio": "innovation_target_ratio",
        "innovation-hidden-dim": "innovation_hidden_dim",
        "innovation-min-keep": "innovation_min_keep",
    }
    for old, new in aliases.items():
        if old in values and new not in values:
            values[new] = values[old]
    return SimpleNamespace(**values)


def load_model_and_testset(
    checkpoint_path: Path,
    cli: argparse.Namespace,
) -> Tuple[torch.nn.Module, torch.utils.data.Dataset, SimpleNamespace, Dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    eval_args = make_eval_args(checkpoint, cli)
    ret = build_dataset(args=eval_args)
    if len(ret) != 4:
        raise RuntimeError("PAMAP2 build_dataset must return train, val, test, num_classes")
    _, _, test_dataset, num_classes = ret
    eval_args.nb_classes = num_classes
    model = build_student_model(eval_args)
    state = checkpoint.get("model", checkpoint)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return model, test_dataset, eval_args, checkpoint


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def state_dict_size_mb(model: torch.nn.Module) -> float:
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as handle:
        path = Path(handle.name)
    try:
        torch.save(model.state_dict(), path)
        return path.stat().st_size / 1024 / 1024
    finally:
        path.unlink(missing_ok=True)


def parameter_bytes(model: torch.nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


class PeakRssMonitor:
    def __init__(self, interval: float = 0.005):
        import psutil

        self.process = psutil.Process(os.getpid())
        self.interval = interval
        self.before = self.process.memory_info().rss
        self.peak = self.before
        self.running = False
        self.thread: Optional[threading.Thread] = None

    def _run(self):
        while self.running:
            self.peak = max(self.peak, self.process.memory_info().rss)
            time.sleep(self.interval)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self) -> Dict[str, float]:
        self.running = False
        if self.thread is not None:
            self.thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)
        mib = 1024 * 1024
        return {
            "process_rss_before_mb": self.before / mib,
            "process_rss_peak_mb": self.peak / mib,
            "process_rss_delta_peak_mb": (self.peak - self.before) / mib,
        }


def safe_logits(output: torch.Tensor) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def routing_network(model: torch.nn.Module):
    return getattr(model, "reservoir_network", None)


def set_routing_collection(model: torch.nn.Module, enabled: bool) -> None:
    network = routing_network(model)
    if network is not None and hasattr(network, "collect_routing_stats"):
        network.collect_routing_stats = enabled


def get_routing_stats(model: torch.nn.Module) -> Dict[str, float]:
    if hasattr(model, "get_routing_stats"):
        return dict(model.get_routing_stats())
    return {}


@torch.inference_mode()
def evaluate_predictions(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
) -> Dict[str, float]:
    model.to(device).eval()
    set_routing_collection(model, True)
    predictions: List[int] = []
    targets: List[int] = []
    routing_rows: List[Dict[str, float]] = []

    for index in range(len(dataset)):
        sample, target = dataset[index]
        if not isinstance(sample, torch.Tensor):
            sample = torch.as_tensor(sample)
        sample = sample.float().unsqueeze(0).to(device)
        output = safe_logits(model(sample))
        predictions.append(int(output.argmax(dim=1).item()))
        targets.append(int(target))
        stats = get_routing_stats(model)
        if stats:
            routing_rows.append(stats)

    result = {
        "accuracy": accuracy_score(targets, predictions),
        "f1_macro": f1_score(targets, predictions, average="macro", zero_division=0),
        "precision_macro": precision_score(
            targets, predictions, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            targets, predictions, average="macro", zero_division=0
        ),
        "num_test_samples": len(targets),
        "routing_route_probability_mean": "",
        "routing_hard_keep_ratio": "",
        "routing_hard_keep_ratio_std": "",
        "routing_selected_patches_mean": "",
        "routing_innovation_distance_mean": "",
        "routing_innovation_cosine_mean": "",
    }
    if routing_rows:
        for key in routing_rows[0]:
            values = [row[key] for row in routing_rows if key in row]
            result[f"routing_{key}"] = float(statistics.fmean(values))
            if key == "hard_keep_ratio":
                result["routing_hard_keep_ratio_std"] = float(
                    statistics.pstdev(values)
                )
    return result


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def representative_samples(
    dataset: torch.utils.data.Dataset,
    count: int,
    device: torch.device,
) -> List[torch.Tensor]:
    if len(dataset) == 0:
        raise RuntimeError("Test dataset is empty")
    count = max(1, min(count, len(dataset)))
    indices = np.linspace(0, len(dataset) - 1, num=count, dtype=int)
    samples: List[torch.Tensor] = []
    for index in indices:
        sample, _ = dataset[int(index)]
        if not isinstance(sample, torch.Tensor):
            sample = torch.as_tensor(sample)
        samples.append(sample.float().unsqueeze(0).to(device))
    return samples


@torch.inference_mode()
def measure_latency(
    model: torch.nn.Module,
    samples: Sequence[torch.Tensor],
    device: torch.device,
    warmup: int,
    repeats: int,
) -> Dict[str, float]:
    model.to(device).eval()
    set_routing_collection(model, False)
    for index in range(warmup):
        safe_logits(model(samples[index % len(samples)]))
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    timings_ms: List[float] = []
    overall_start = time.perf_counter()
    total_calls = 0
    for _ in range(repeats):
        for sample in samples:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter_ns()
            safe_logits(model(sample))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            end = time.perf_counter_ns()
            timings_ms.append((end - start) / 1e6)
            total_calls += 1
    total_seconds = time.perf_counter() - overall_start
    mean_ms = statistics.fmean(timings_ms)
    return {
        "latency_mean_ms": mean_ms,
        "latency_std_ms": statistics.pstdev(timings_ms),
        "latency_p50_ms": percentile(timings_ms, 50),
        "latency_p95_ms": percentile(timings_ms, 95),
        "latency_p99_ms": percentile(timings_ms, 99),
        "throughput_samples_per_sec": total_calls / total_seconds,
        "latency_measurements": len(timings_ms),
    }


@torch.inference_mode()
def measure_peak_memory(
    model: torch.nn.Module,
    samples: Sequence[torch.Tensor],
    device: torch.device,
) -> Dict[str, float]:
    model.to(device).eval()
    set_routing_collection(model, False)
    try:
        monitor = PeakRssMonitor()
    except ImportError:
        monitor = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    if monitor is not None:
        monitor.start()
    for sample in samples:
        safe_logits(model(sample))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = monitor.stop() if monitor is not None else {}
    if device.type == "cuda":
        result.update(
            {
                "gpu_peak_allocated_mb": torch.cuda.max_memory_allocated(device)
                / 1024
                / 1024,
                "gpu_peak_reserved_mb": torch.cuda.max_memory_reserved(device)
                / 1024
                / 1024,
            }
        )
    return result


def measure_thop(model: torch.nn.Module, sample: torch.Tensor) -> Dict:
    """Compatibility metric only; THOP misses custom torch.matmul operations."""
    try:
        from thop import profile
    except ImportError:
        return {"thop_macs": "", "thop_flops_2x_macs": "", "thop_params": ""}
    set_routing_collection(model, False)
    macs, params = profile(model, inputs=(sample,), verbose=False)
    return {
        "thop_macs": float(macs),
        "thop_flops_2x_macs": float(2 * macs),
        "thop_params": float(params),
    }


def num_patches(signal_length: int, patch_size: int, stride: Optional[int] = None) -> int:
    stride = patch_size if stride is None else stride
    return math.floor((signal_length - patch_size) / stride) + 1


def theoretical_reservoir_macs(eval_args: SimpleNamespace, metrics: Dict) -> Dict:
    model = eval_args.student
    if model not in {"PRC", "PRC_LRGR", "APS_LRGR", "SIR_LRGR"}:
        return {
            "num_patches": "",
            "selected_patches_mean": "",
            "actual_keep_ratio": "",
            "theoretical_reservoir_macs": "",
            "theoretical_total_macs": "",
        }
    length = int(eval_args.input_size)
    patch = int(eval_args.patch_size)
    stride = patch
    channels = 3
    classes = int(eval_args.nb_classes)
    reservoir = int(eval_args.reservoir_size)
    rank = int(getattr(eval_args, "reservoir_rank", 0))
    patches = num_patches(length, patch, stride)
    conv_macs = patches * patch * channels * patch
    head_macs = 2 * reservoir * classes

    selected = float(patches)
    if model == "PRC":
        updates = patches + 2  # normal patches plus CLS and DIST token updates
        reservoir_macs = updates * (reservoir * reservoir + patch * reservoir)
    else:
        if model == "APS_LRGR":
            keep = math.ceil(patches * float(eval_args.patch_keep_ratio))
        elif model == "SIR_LRGR":
            ratio = metrics.get("routing_hard_keep_ratio", float("nan"))
            keep = patches if not math.isfinite(float(ratio)) else patches * float(ratio)
        else:
            keep = patches
        selected = float(keep)
        patch_update = 2 * reservoir * rank + patch * reservoir + patch
        token_update = 2 * (reservoir * rank + patch * reservoir)
        reservoir_macs = keep * patch_update + token_update
        if model == "APS_LRGR":
            reservoir_macs += patches * patch  # patch scorer
        if model == "SIR_LRGR":
            hidden = int(getattr(eval_args, "innovation_hidden_dim", 64))
            router_per_patch = patch * rank + 4 * rank * hidden + hidden
            reservoir_macs += patches * router_per_patch
    return {
        "num_patches": patches,
        "selected_patches_mean": selected,
        "actual_keep_ratio": selected / patches,
        "theoretical_reservoir_macs": float(reservoir_macs),
        "theoretical_total_macs": float(conv_macs + reservoir_macs + head_macs),
    }


@dataclass
class EnergyResult:
    backend: str = ""
    gross_energy_j: float = float("nan")
    idle_power_w: float = float("nan")
    average_power_w: float = float("nan")
    net_energy_j: float = float("nan")
    duration_s: float = float("nan")
    samples: int = 0


class RaplMeter:
    def __init__(self):
        self.use_sudo = os.environ.get("RAPL_USE_SUDO", "0") == "1"
        roots = list(Path("/sys/class/powercap").glob("intel-rapl:*"))
        self.zones: List[Tuple[Path, Optional[int]]] = []
        for root in roots:
            energy = root / "energy_uj"
            name = root / "name"
            if not energy.exists():
                continue
            zone_name = self._read_text(name) if name.exists() else ""
            if zone_name.startswith("package-") or zone_name.startswith("package"):
                maximum = root / "max_energy_range_uj"
                max_value = int(self._read_text(maximum)) if maximum.exists() else None
                self.zones.append((energy, max_value))
        if not self.zones:
            raise RuntimeError("Package-level Linux RAPL counters are unavailable")
        # Fail before checkpoint evaluation if the counters are not readable.
        self.read()

    def _read_text(self, path: Path) -> str:
        if self.use_sudo:
            completed = subprocess.run(
                ["sudo", "-n", "cat", str(path)],
                check=True,
                capture_output=True,
                text=True,
            )
            return completed.stdout.strip()
        return path.read_text().strip()

    def read(self) -> List[int]:
        return [int(self._read_text(path)) for path, _ in self.zones]

    def delta_j(self, start: List[int], end: List[int]) -> float:
        total = 0
        for index, (before, after) in enumerate(zip(start, end)):
            maximum = self.zones[index][1]
            delta = after - before
            if delta < 0 and maximum:
                delta += maximum
            total += delta
        return total / 1e6


class NvmlMeter:
    def __init__(self, interval: float):
        import pynvml

        self.nvml = pynvml
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.interval = interval
        self.cumulative_supported = True
        try:
            pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
        except Exception:
            self.cumulative_supported = False

    def read_energy_j(self) -> float:
        return self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle) / 1000.0

    def power_w(self) -> float:
        return self.nvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0


class PowerSampler:
    def __init__(self, meter: NvmlMeter):
        self.meter = meter
        self.samples: List[Tuple[float, float]] = []
        self.running = False
        self.thread: Optional[threading.Thread] = None

    def _run(self):
        while self.running:
            self.samples.append((time.perf_counter(), self.meter.power_w()))
            time.sleep(self.meter.interval)

    def start(self):
        self.samples = []
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self) -> float:
        self.running = False
        if self.thread is not None:
            self.thread.join()
        if len(self.samples) < 2:
            return float("nan")
        energy = 0.0
        for (t0, p0), (t1, p1) in zip(self.samples, self.samples[1:]):
            energy += 0.5 * (p0 + p1) * (t1 - t0)
        return energy


def select_energy_meter(device: torch.device, interval: float):
    if device.type == "cuda":
        try:
            return "nvml", NvmlMeter(interval)
        except Exception as error:
            print(f"[warn] NVML energy unavailable: {error}")
            return "", None
    try:
        return "rapl", RaplMeter()
    except Exception as error:
        print(f"[warn] RAPL energy unavailable: {error}")
        return "", None


def measure_idle_energy(backend: str, meter, seconds: float) -> float:
    if backend == "rapl":
        start = meter.read()
        began = time.perf_counter()
        time.sleep(seconds)
        duration = time.perf_counter() - began
        energy = meter.delta_j(start, meter.read())
        return energy / duration
    if backend == "nvml" and meter.cumulative_supported:
        start = meter.read_energy_j()
        began = time.perf_counter()
        time.sleep(seconds)
        duration = time.perf_counter() - began
        return (meter.read_energy_j() - start) / duration
    if backend == "nvml":
        sampler = PowerSampler(meter)
        began = time.perf_counter()
        sampler.start()
        time.sleep(seconds)
        energy = sampler.stop()
        duration = time.perf_counter() - began
        return energy / duration
    return float("nan")


@torch.inference_mode()
def measure_energy(
    model: torch.nn.Module,
    samples: Sequence[torch.Tensor],
    device: torch.device,
    seconds: float,
    idle_seconds: float,
    interval: float,
) -> EnergyResult:
    backend, meter = select_energy_meter(device, interval)
    if meter is None:
        return EnergyResult()
    model.to(device).eval()
    set_routing_collection(model, False)
    idle_power = measure_idle_energy(backend, meter, idle_seconds)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    sampler = None
    if backend == "rapl":
        start_value = meter.read()
    elif meter.cumulative_supported:
        start_value = meter.read_energy_j()
    else:
        sampler = PowerSampler(meter)
        sampler.start()
        start_value = None

    began = time.perf_counter()
    calls = 0
    while time.perf_counter() - began < seconds:
        sample = samples[calls % len(samples)]
        safe_logits(model(sample))
        calls += 1
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    duration = time.perf_counter() - began

    if backend == "rapl":
        gross = meter.delta_j(start_value, meter.read())
    elif meter.cumulative_supported:
        gross = meter.read_energy_j() - start_value
    else:
        gross = sampler.stop() if sampler is not None else float("nan")
    average_power = gross / duration
    net = max(0.0, gross - idle_power * duration)
    return EnergyResult(
        backend=backend,
        gross_energy_j=gross,
        idle_power_w=idle_power,
        average_power_w=average_power,
        net_energy_j=net,
        duration_s=duration,
        samples=calls,
    )


def energy_to_dict(result: EnergyResult, accuracy: float) -> Dict:
    if not result.backend or result.samples <= 0:
        return {
            "energy_backend": "",
            "energy_duration_s": "",
            "energy_samples": "",
            "idle_power_w": "",
            "average_power_w": "",
            "gross_energy_j": "",
            "net_energy_j": "",
            "gross_joules_per_sample": "",
            "net_joules_per_sample": "",
            "samples_per_gross_joule": "",
            "correct_predictions_per_gross_joule": "",
        }
    gross_jps = result.gross_energy_j / result.samples
    net_jps = result.net_energy_j / result.samples
    return {
        "energy_backend": result.backend,
        "energy_duration_s": result.duration_s,
        "energy_samples": result.samples,
        "idle_power_w": result.idle_power_w,
        "average_power_w": result.average_power_w,
        "gross_energy_j": result.gross_energy_j,
        "net_energy_j": result.net_energy_j,
        "gross_joules_per_sample": gross_jps,
        "net_joules_per_sample": net_jps,
        "samples_per_gross_joule": 1.0 / gross_jps if gross_jps > 0 else "",
        "correct_predictions_per_gross_joule": (
            accuracy / gross_jps if gross_jps > 0 else ""
        ),
    }


def append_row(path: Path, row: Dict, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite or not path.exists():
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        return
    with path.open("r", newline="", encoding="utf-8") as handle:
        existing_header = next(csv.reader(handle), None)
    if existing_header != list(row.keys()):
        raise RuntimeError(
            "Output CSV header differs from this script version. Use a new file or --overwrite."
        )
    with path.open("a", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=list(row.keys())).writerow(row)


def read_csv_rows(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


SUMMARY_METRICS = (
    "accuracy",
    "f1_macro",
    "precision_macro",
    "recall_macro",
    "params_total",
    "params_trainable",
    "parameter_bytes",
    "parameter_memory_mb",
    "model_size_mb",
    "num_patches",
    "selected_patches_mean",
    "actual_keep_ratio",
    "theoretical_reservoir_macs",
    "theoretical_total_macs",
    "thop_macs",
    "thop_flops_2x_macs",
    "thop_params",
    "routing_route_probability_mean",
    "routing_hard_keep_ratio",
    "routing_hard_keep_ratio_std",
    "routing_selected_patches_mean",
    "routing_innovation_distance_mean",
    "routing_innovation_cosine_mean",
    "latency_mean_ms",
    "latency_std_ms",
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_p99_ms",
    "throughput_samples_per_sec",
    "process_rss_before_mb",
    "process_rss_peak_mb",
    "process_rss_delta_peak_mb",
    "gpu_peak_allocated_mb",
    "gpu_peak_reserved_mb",
    "idle_power_w",
    "average_power_w",
    "gross_energy_j",
    "net_energy_j",
    "gross_joules_per_sample",
    "net_joules_per_sample",
    "samples_per_gross_joule",
    "correct_predictions_per_gross_joule",
)


def numeric_values(rows: Sequence[Dict], key: str) -> List[float]:
    values: List[float] = []
    for row in rows:
        value = row.get(key, "")
        if value in ("", None):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            values.append(number)
    return values


def write_subject_summary(path: Path, rows: Sequence[Dict]) -> None:
    """Write one row per experiment/model/device with subject mean and sample SD."""
    groups: Dict[Tuple[str, str, str], List[Dict]] = {}
    for row in rows:
        key = (
            str(row.get("experiment_type", "")),
            str(row.get("model_key", "")),
            str(row.get("device", "")),
        )
        groups.setdefault(key, []).append(row)

    summaries: List[Dict] = []
    for key in sorted(groups):
        group = groups[key]
        first = group[0]
        subjects = sorted({int(row["test_subject"]) for row in group})
        summary = {
            "summarized_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "experiment_type": first.get("experiment_type", ""),
            "model_key": first.get("model_key", ""),
            "model": first.get("model", ""),
            "dataset": first.get("dataset", ""),
            "device": first.get("device", ""),
            "platform": first.get("platform", ""),
            "cpu": first.get("cpu", ""),
            "gpu": first.get("gpu", ""),
            "input_shape": first.get("input_shape", ""),
            "patch_size": first.get("patch_size", ""),
            "reservoir_size": first.get("reservoir_size", ""),
            "reservoir_rank": first.get("reservoir_rank", ""),
            "patch_keep_ratio": first.get("patch_keep_ratio", ""),
            "innovation_target_ratio": first.get("innovation_target_ratio", ""),
            "innovation_threshold": first.get("innovation_threshold", ""),
            "innovation_hidden_dim": first.get("innovation_hidden_dim", ""),
            "innovation_min_keep": first.get("innovation_min_keep", ""),
            "energy_backend": first.get("energy_backend", ""),
            "n_checkpoints": len(group),
            "n_subjects": len(subjects),
            "test_subjects": ";".join(str(subject) for subject in subjects),
            "num_test_samples_total": int(
                sum(numeric_values(group, "num_test_samples"))
            ),
        }

        sample_counts = numeric_values(group, "num_test_samples")
        accuracies = numeric_values(group, "accuracy")
        if len(sample_counts) == len(accuracies) and sum(sample_counts) > 0:
            summary["accuracy_sample_weighted"] = sum(
                row_accuracy * count
                for row_accuracy, count in zip(accuracies, sample_counts)
            ) / sum(sample_counts)
        else:
            summary["accuracy_sample_weighted"] = ""

        for metric in SUMMARY_METRICS:
            values = numeric_values(group, metric)
            summary[f"{metric}_subject_mean"] = (
                float(statistics.fmean(values)) if values else ""
            )
            summary[f"{metric}_subject_std"] = (
                float(statistics.stdev(values)) if len(values) >= 2 else 0.0 if values else ""
            )
        summaries.append(summary)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)


def format_mean_std(values: Sequence[float], scale: float = 1.0) -> str:
    if not values:
        return ""
    mean = statistics.fmean(values) * scale
    std = statistics.stdev(values) * scale if len(values) >= 2 else 0.0
    return f"{mean:.2f} ({std:.2f})"


def format_mean(values: Sequence[float], scale: float = 1.0, digits: int = 3) -> str:
    if not values:
        return ""
    return f"{statistics.fmean(values) * scale:.{digits}f}"


def paper_model_name(row: Dict) -> str:
    names = {
        "PRC": "PRC",
        "PRC_LRGR": "LRGR",
        "APS_LRGR": "APS-LRGR",
        "SIR_LRGR": "SIR-LRGR",
    }
    method = names.get(str(row.get("model", "")), str(row.get("model", "")))
    return f"{method}-p{row.get('patch_size', '')}"


def write_word_comparison(path: Path, rows: Sequence[Dict]) -> None:
    """Create one compact row per model configuration for the Word paper table."""
    groups: Dict[Tuple[str, str], List[Dict]] = {}
    for row in rows:
        key = (str(row.get("model_key", "")), str(row.get("device", "")))
        groups.setdefault(key, []).append(row)

    method_order = {"PRC": 0, "PRC_LRGR": 1, "APS_LRGR": 2, "SIR_LRGR": 3}
    table_rows: List[Tuple[Tuple[int, int], Dict]] = []
    for group in groups.values():
        first = group[0]
        baseline = [row for row in group if row.get("experiment_type") == "baseline"]
        transfer = [row for row in group if row.get("experiment_type") == "transfer"]
        efficiency = transfer if transfer else baseline

        baseline_accuracy = numeric_values(baseline, "accuracy")
        transfer_accuracy = numeric_values(transfer, "accuracy")
        baseline_f1 = numeric_values(baseline, "f1_macro")
        transfer_f1 = numeric_values(transfer, "f1_macro")

        baseline_by_subject = {
            int(row["test_subject"]): float(row["accuracy"])
            for row in baseline
            if row.get("accuracy", "") not in ("", None)
        }
        transfer_by_subject = {
            int(row["test_subject"]): float(row["accuracy"])
            for row in transfer
            if row.get("accuracy", "") not in ("", None)
        }
        common_subjects = sorted(set(baseline_by_subject) & set(transfer_by_subject))
        paired_deltas = [
            100.0 * (transfer_by_subject[subject] - baseline_by_subject[subject])
            for subject in common_subjects
        ]

        table_row = {
            "Model": paper_model_name(first),
            "Model key": first.get("model_key", ""),
            "Device": first.get("device", ""),
            "Base Acc [%]": format_mean_std(baseline_accuracy, scale=100.0),
            "Transfer Acc [%]": format_mean_std(transfer_accuracy, scale=100.0),
            "Delta [pt]": format_mean(paired_deltas, digits=2),
            "Base macro F1 [%]": format_mean_std(baseline_f1, scale=100.0),
            "Transfer macro F1 [%]": format_mean_std(transfer_f1, scale=100.0),
            "Size [MB]": format_mean(numeric_values(efficiency, "model_size_mb")),
            "Parameter memory [MB]": format_mean(
                numeric_values(efficiency, "parameter_memory_mb")
            ),
            "Theoretical MACs [M]": format_mean(
                numeric_values(efficiency, "theoretical_total_macs"), scale=1e-6
            ),
            "Latency [ms]": format_mean(
                numeric_values(efficiency, "latency_mean_ms"), digits=2
            ),
            "Keep ratio [%]": format_mean(
                numeric_values(efficiency, "actual_keep_ratio"), scale=100.0, digits=2
            ),
            "Gross energy [mJ/sample]": format_mean(
                numeric_values(efficiency, "gross_joules_per_sample"),
                scale=1000.0,
                digits=4,
            ),
            "Net energy [mJ/sample]": format_mean(
                numeric_values(efficiency, "net_joules_per_sample"),
                scale=1000.0,
                digits=4,
            ),
            "Baseline subjects": len({row["test_subject"] for row in baseline}),
            "Transfer subjects": len({row["test_subject"] for row in transfer}),
            "Paired subjects": len(common_subjects),
        }
        sort_key = (
            int(first.get("patch_size", 0)),
            method_order.get(str(first.get("model", "")), 99),
        )
        table_rows.append((sort_key, table_row))

    ordered_rows = [row for _, row in sorted(table_rows, key=lambda item: item[0])]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ordered_rows[0].keys()))
        writer.writeheader()
        writer.writerows(ordered_rows)


def evaluate_one(path: Path, cli: argparse.Namespace, device: torch.device) -> Dict:
    print(f"\n===== evaluating: {path} =====")
    model, test_dataset, eval_args, checkpoint = load_model_and_testset(path, cli)
    total_params, trainable_params = count_parameters(model)
    size_mb = state_dict_size_mb(model)
    metrics = evaluate_predictions(model, test_dataset, device)
    samples = representative_samples(
        test_dataset,
        count=max(1, cli.latency_samples),
        device=device,
    )
    latency = {}
    if not cli.skip_latency:
        latency = measure_latency(
            model,
            samples,
            device,
            warmup=cli.warmup,
            repeats=cli.latency_repeats,
        )
    memory = {} if cli.skip_memory else measure_peak_memory(model, samples, device)
    thop = {} if cli.skip_thop else measure_thop(model, samples[0])
    energy_result = EnergyResult()
    if cli.measure_energy:
        energy_result = measure_energy(
            model,
            samples,
            device,
            seconds=cli.energy_seconds,
            idle_seconds=cli.idle_seconds,
            interval=cli.power_sample_interval,
        )
    theoretical = theoretical_reservoir_macs(eval_args, metrics)
    row = {
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "experiment_type": experiment_type(path),
        "model_key": model_key_from_checkpoint(path),
        "model": eval_args.student,
        "dataset": "PAMAP2",
        "test_subject": int(eval_args.fold_test_subj),
        "val_subject": int(eval_args.fold_val_subj),
        "checkpoint": str(path),
        "checkpoint_sha256": checkpoint_sha256(path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "device": str(device),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cpu": platform.processor(),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "",
        "input_shape": f"(1, 3, {eval_args.input_size})",
        "patch_size": int(eval_args.patch_size),
        "reservoir_size": int(eval_args.reservoir_size),
        "reservoir_rank": int(getattr(eval_args, "reservoir_rank", 0)),
        "patch_keep_ratio": float(getattr(eval_args, "patch_keep_ratio", 1.0)),
        "innovation_target_ratio": float(
            getattr(eval_args, "innovation_target_ratio", 0.0)
        ),
        "innovation_threshold": float(
            getattr(eval_args, "innovation_threshold", 0.0)
        ),
        "innovation_hidden_dim": int(
            getattr(eval_args, "innovation_hidden_dim", 0)
        ),
        "innovation_min_keep": int(
            getattr(eval_args, "innovation_min_keep", 0)
        ),
        "params_total": total_params,
        "params_trainable": trainable_params,
        "parameter_bytes": parameter_bytes(model),
        "parameter_memory_mb": parameter_bytes(model) / 1024 / 1024,
        "model_size_mb": size_mb,
        **theoretical,
        **thop,
        **metrics,
        **latency,
        **memory,
        **energy_to_dict(energy_result, metrics["accuracy"]),
    }
    return row


def main() -> None:
    cli = parse_args()
    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)
    if cli.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    device = torch.device("cuda:0" if cli.device == "cuda" else "cpu")
    checkpoints = find_checkpoints(cli)
    if not checkpoints:
        raise RuntimeError("No best_checkpoint.pth files matched the supplied roots/filters")
    output = Path(cli.output_csv)
    if cli.overwrite and cli.resume_output:
        raise ValueError("--overwrite and --resume-output cannot be used together")
    existing_rows = read_csv_rows(output) if cli.resume_output else []
    existing_checkpoints = {
        str(row.get("checkpoint", ""))
        for row in existing_rows
        if row.get("checkpoint", "")
    }
    if cli.resume_output:
        checkpoints = [
            checkpoint
            for checkpoint in checkpoints
            if str(checkpoint) not in existing_checkpoints
        ]
    if cli.overwrite and output.exists():
        output.unlink()
    print(f"[info] existing rows: {len(existing_rows)}")
    print(f"[info] remaining checkpoints: {len(checkpoints)}")
    print(f"[info] output: {output}")
    first = not output.exists()
    completed_rows = list(existing_rows)
    failures = []
    for checkpoint in checkpoints:
        try:
            row = evaluate_one(checkpoint, cli, device)
            append_row(output, row, overwrite=first and cli.overwrite)
            first = False
            completed_rows.append(row)
            print(
                f"[done] {row['model_key']} test{row['test_subject']} "
                f"acc={row['accuracy']:.4f} f1={row['f1_macro']:.4f}"
            )
        except Exception as error:
            failures.append((str(checkpoint), repr(error)))
            print(f"[error] {checkpoint}: {error}")
        finally:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    if completed_rows:
        summary_output = (
            Path(cli.summary_csv)
            if cli.summary_csv
            else output.with_name(f"{output.stem}_summary.csv")
        )
        write_subject_summary(summary_output, completed_rows)
        print(f"[done] subject summary: {summary_output}")
        if cli.word_table_csv:
            word_table_output = Path(cli.word_table_csv)
            write_word_comparison(word_table_output, completed_rows)
            print(f"[done] Word table: {word_table_output}")
    if failures:
        failure_path = output.with_suffix(".failures.json")
        failure_path.write_text(
            json.dumps(
                [{"checkpoint": path, "error": error} for path, error in failures],
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"[warn] failures: {len(failures)} -> {failure_path}")


if __name__ == "__main__":
    main()
