import argparse
import csv
import gc
import os
import shutil
import tempfile
import threading
import time
import tracemalloc
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False

import models.MLPMixer4sig_dist as MLPMixer4sig_dist
import models.PatchEchoClassifier as PatchEchoClassifier
import models.PatchEchosAttnClassifier as PatchEchosAttnClassifier
import models.DeepConvLSTM as DeepConvLSTM
import models.resnet4sig as resnet4sig

try:
    import models.PatchEchoAttnClassifier as PatchEchoAttnClassifier
except Exception:
    PatchEchoAttnClassifier = None

try:
    from models import PatchLowRankGatedReservoir
except Exception:
    PatchLowRankGatedReservoir = None

try:
    from models import PatchAdaptiveLowRankGatedReservoir
except Exception:
    PatchAdaptiveLowRankGatedReservoir = None

try:
    from models import PatchInputFactorizedStateAttentiveLRGR
except Exception:
    PatchInputFactorizedStateAttentiveLRGR = None

try:
    from models import PatchStateInnovationLRGR
except Exception:
    PatchStateInnovationLRGR = None

try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    profile = None
    THOP_AVAILABLE = False


def get_args():
    parser = argparse.ArgumentParser(
        description="Measure model footprint for sensor HAR models."
    )

    parser.add_argument("--dataset", type=str, default="SHL2023")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--run-name", type=str, default=None)

    parser.add_argument(
        "--input-size",
        "--signal-length",
        dest="signal_length",
        type=int,
        default=496
    )

    parser.add_argument("--num-classes", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--channels", type=int, default=3)

    parser.add_argument(
        "--patch_size",
        "--patch-size",
        dest="patch_size",
        type=int,
        default=16
    )
    parser.add_argument(
        "--reservoir_size",
        "--reservoir-size",
        dest="reservoir_size",
        type=int,
        default=1000
    )
    parser.add_argument(
        "--reservoir_rank",
        "--reservoir-rank",
        dest="reservoir_rank",
        type=int,
        default=64
    )
    parser.add_argument(
        "--patch_keep_ratio",
        "--patch-keep-ratio",
        dest="patch_keep_ratio",
        type=float,
        default=0.5
    )
    parser.add_argument(
        "--input_rank",
        "--input-rank",
        dest="input_rank",
        type=int,
        default=16
    )

    parser.add_argument(
        "--innovation_threshold",
        "--innovation-threshold",
        dest="innovation_threshold",
        type=float,
        default=0.5
    )
    
    parser.add_argument(
        "--innovation_target_ratio",
        "--innovation-target-ratio",
        dest="innovation_target_ratio",
        type=float,
        default=0.5
    )
    
    parser.add_argument(
        "--innovation_hidden_dim",
        "--innovation-hidden-dim",
        dest="innovation_hidden_dim",
        type=int,
        default=64
    )
    
    parser.add_argument(
        "--innovation_min_keep",
        "--innovation-min-keep",
        dest="innovation_min_keep",
        type=int,
        default=1
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"]
    )

    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeat", type=int, default=200)

    parser.add_argument(
        "--memory-warmup",
        type=int,
        default=10,
        help="Warmup count for inference memory measurement."
    )
    parser.add_argument(
        "--memory-repeat",
        type=int,
        default=None,
        help="Repeat count for inference memory measurement. If None, use --repeat."
    )
    parser.add_argument(
        "--ram-monitor-interval",
        type=float,
        default=0.01,
        help="Interval seconds for peak RAM monitoring."
    )
    parser.add_argument(
        "--skip-memory",
        action="store_true",
        help="Skip RAM/heap/GPU memory measurement."
    )

    parser.add_argument(
        "--csv-path",
        type=str,
        default="experiments/SHL2023_senvt_echo_w496/footprint/model_static_metrics.csv"
    )

    return parser.parse_args()


def make_model_key(args) -> str:
    if args.run_name is not None and args.run_name != "":
        return args.run_name

    if args.model in ["PRC", "PESAC", "PEAC", "PEAC_Q"]:
        return f"{args.model}_p{args.patch_size}_r{args.reservoir_size}"

    if args.model == "PRC_LRGR":
        return (
            f"LRGR_p{args.patch_size}"
            f"_r{args.reservoir_size}"
            f"_rank{args.reservoir_rank}"
        )

    if args.model == "APS_LRGR":
        keep = int(round(args.patch_keep_ratio * 100))
        return (
            f"APS_LRGR_p{args.patch_size}"
            f"_r{args.reservoir_size}"
            f"_rank{args.reservoir_rank}"
            f"_keep{keep:03d}"
        )

    if args.model == "IFSA_LRGR":
        return (
            f"IFSA_LRGR_p{args.patch_size}"
            f"_r{args.reservoir_size}"
            f"_rank{args.reservoir_rank}"
            f"_irank{args.input_rank}"
        )

    if args.model == "SIR_LRGR":
        target = int(round(args.innovation_target_ratio * 100))
    
        return (
            f"SIR_LRGR_p{args.patch_size}"
            f"_r{args.reservoir_size}"
            f"_rank{args.reservoir_rank}"
            f"_target{target:03d}"
        )
    return args.model


def build_deepconvlstm_config(model_name: str, input_size: int, num_classes: int) -> Dict:
    if "100" in model_name:
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

    if "50" in model_name:
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

    if "25" in model_name:
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

    raise ValueError(f"Unknown DeepConvLSTM variant: {model_name}")


def build_resnet(model_name: str, num_classes: int) -> nn.Module:
    if "L" in model_name:
        return resnet4sig.ResNet1D(
            in_channels=3,
            base_filters=64,
            kernel_size=7,
            stride=2,
            groups=1,
            n_block=8,
            n_classes=num_classes,
            downsample_gap=2,
            increasefilter_gap=4,
            use_bn=True,
            use_do=True,
            verbose=False,
        )

    if "M" in model_name:
        return resnet4sig.ResNet1D(
            in_channels=3,
            base_filters=32,
            kernel_size=7,
            stride=2,
            groups=1,
            n_block=8,
            n_classes=num_classes,
            downsample_gap=2,
            increasefilter_gap=4,
            use_bn=True,
            use_do=True,
            verbose=False,
        )

    if "S" in model_name:
        return resnet4sig.ResNet1D(
            in_channels=3,
            base_filters=16,
            kernel_size=7,
            stride=2,
            groups=1,
            n_block=4,
            n_classes=num_classes,
            downsample_gap=2,
            increasefilter_gap=4,
            use_bn=True,
            use_do=True,
            verbose=False,
        )

    raise ValueError(f"Unknown Resnet variant: {model_name}")


def build_model(args) -> nn.Module:
    if args.model == "MLPMixer":
        model = MLPMixer4sig_dist.DistilledMLPMixer(
            dim=512,
            num_classes=args.num_classes,
            depth=8
        )

    elif args.model == "PRC":
        model = PatchEchoClassifier.PatchReservoir(
            in_channels=args.channels,
            patch_size=args.patch_size,
            stride=args.patch_size,
            reservoir_size=args.reservoir_size,
            num_classes=args.num_classes
        )

    elif args.model == "PESAC":
        model = PatchEchosAttnClassifier.PatchReservoir(
            in_channels=args.channels,
            patch_size=args.patch_size,
            stride=args.patch_size,
            reservoir_size=args.reservoir_size,
            num_classes=args.num_classes
        )

    elif args.model == "PEAC":
        if PatchEchoAttnClassifier is None:
            raise ImportError(
                "models/PatchEchoAttnClassifier.py が import できません。"
                "PEACを使わないならこの分岐は無視してOKです。"
            )

        model = PatchEchoAttnClassifier.PatchReservoir(
            in_channels=args.channels,
            patch_size=args.patch_size,
            stride=args.patch_size,
            reservoir_size=args.reservoir_size,
            num_classes=args.num_classes
        )

    elif args.model == "PEAC_Q":
        if PatchEchoAttnClassifier is None:
            raise ImportError(
                "models/PatchEchoAttnClassifier.py が import できません。"
                "PEAC_Qを使わないならこの分岐は無視してOKです。"
            )

        model = PatchEchoAttnClassifier.PatchReservoir(
            in_channels=args.channels,
            patch_size=args.patch_size,
            stride=args.patch_size,
            reservoir_size=args.reservoir_size,
            num_classes=args.num_classes
        )

        model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )

    elif args.model == "PRC_LRGR":
        if PatchLowRankGatedReservoir is None:
            raise ImportError(
                "models/PatchLowRankGatedReservoir.py が import できません。"
                "先に LRGR のモデルファイルを作成してください。"
            )

        model = PatchLowRankGatedReservoir.PatchReservoir(
            in_channels=args.channels,
            patch_size=args.patch_size,
            stride=args.patch_size,
            reservoir_size=args.reservoir_size,
            reservoir_rank=args.reservoir_rank,
            num_classes=args.num_classes
        )

    elif args.model == "APS_LRGR":
        if PatchAdaptiveLowRankGatedReservoir is None:
            raise ImportError(
                "models/PatchAdaptiveLowRankGatedReservoir.py が import できません。"
                "先に APS-LRGR のモデルファイルを作成してください。"
            )

        model = PatchAdaptiveLowRankGatedReservoir.PatchReservoir(
            in_channels=args.channels,
            patch_size=args.patch_size,
            stride=args.patch_size,
            reservoir_size=args.reservoir_size,
            reservoir_rank=args.reservoir_rank,
            patch_keep_ratio=args.patch_keep_ratio,
            num_classes=args.num_classes
        )

    elif args.model == "IFSA_LRGR":
        if PatchInputFactorizedStateAttentiveLRGR is None:
            raise ImportError(
                "models/PatchInputFactorizedStateAttentiveLRGR.py が import できません。"
                "先に IFSA-LRGR のモデルファイルを作成してください。"
            )

        model = PatchInputFactorizedStateAttentiveLRGR.PatchReservoir(
            in_channels=args.channels,
            patch_size=args.patch_size,
            stride=args.patch_size,
            reservoir_size=args.reservoir_size,
            reservoir_rank=args.reservoir_rank,
            input_rank=args.input_rank,
            num_classes=args.num_classes
        )

    elif args.model == "SIR_LRGR":
        if PatchStateInnovationLRGR is None:
            raise ImportError(
                "models/PatchStateInnovationLRGR.py が import できません。"
            )
    
        model = PatchStateInnovationLRGR.PatchReservoir(
            in_channels=args.channels,
            patch_size=args.patch_size,
            stride=args.patch_size,
            reservoir_size=args.reservoir_size,
            reservoir_rank=args.reservoir_rank,
            num_classes=args.num_classes,
            router_hidden_dim=args.innovation_hidden_dim,
            routing_threshold=args.innovation_threshold,
            target_keep_ratio=args.innovation_target_ratio,
            minimum_keep_patches=args.innovation_min_keep,
            collect_routing_stats=False,
        )

    elif "DeepConvLSTM" in args.model:
        config = build_deepconvlstm_config(
            model_name=args.model,
            input_size=args.signal_length,
            num_classes=args.num_classes
        )
        model = DeepConvLSTM.DeepConvLSTM(**config)

    elif "Resnet" in args.model:
        model = build_resnet(
            model_name=args.model,
            num_classes=args.num_classes
        )

    else:
        raise ValueError(f"Unknown model: {args.model}")

    return model


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def estimate_model_size_mb(model: nn.Module) -> float:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as f:
        tmp_path = f.name

    try:
        torch.save(model.state_dict(), tmp_path)
        size_mb = os.path.getsize(tmp_path) / 1024 / 1024
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return size_mb


@torch.no_grad()
def safe_forward(model: nn.Module, x: torch.Tensor):
    out = model(x)

    if isinstance(out, (tuple, list)):
        return out[0]

    return out


def measure_flops(model: nn.Module, x: torch.Tensor) -> Tuple[Optional[float], Optional[float]]:
    if not THOP_AVAILABLE:
        print("[warn] thop is not installed. FLOPs will be empty.")
        return None, None

    model.eval()

    try:
        flops, params = profile(
            model,
            inputs=(x,),
            verbose=False
        )
        return float(flops), float(params)

    except Exception as e:
        print(f"[warn] FLOPs measurement failed: {e}")
        return None, None


def measure_latency(
    model: nn.Module,
    x: torch.Tensor,
    device: str = "cpu",
    warmup: int = 30,
    repeat: int = 200
) -> Tuple[float, float, float]:
    model.to(device)
    x = x.to(device)
    model.eval()

    with torch.no_grad():
        for _ in range(warmup):
            _ = safe_forward(model, x)

        if device.startswith("cuda"):
            torch.cuda.synchronize()

        start = time.perf_counter()

        for _ in range(repeat):
            _ = safe_forward(model, x)

        if device.startswith("cuda"):
            torch.cuda.synchronize()

        end = time.perf_counter()

    avg_batch_latency = (end - start) / repeat
    avg_sample_latency = avg_batch_latency / x.size(0)
    throughput = x.size(0) / avg_batch_latency

    return avg_batch_latency, avg_sample_latency, throughput


class PeakRAMMonitor:
    def __init__(self, interval: float = 0.01):
        self.interval = interval
        self.peak_mb = ""
        self.running = False
        self.thread = None

        if PSUTIL_AVAILABLE:
            self.process = psutil.Process(os.getpid())
        else:
            self.process = None

    def _monitor(self):
        while self.running:
            rss_mb = self.process.memory_info().rss / 1024 / 1024
            if self.peak_mb == "" or rss_mb > self.peak_mb:
                self.peak_mb = rss_mb
            time.sleep(self.interval)

    def start(self):
        if not PSUTIL_AVAILABLE:
            self.peak_mb = ""
            return

        self.peak_mb = self.process.memory_info().rss / 1024 / 1024
        self.running = True
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def stop(self):
        if not PSUTIL_AVAILABLE:
            return ""

        self.running = False
        if self.thread is not None:
            self.thread.join()

        return self.peak_mb


def measure_inference_memory(
    model: nn.Module,
    x: torch.Tensor,
    device: str = "cpu",
    warmup: int = 10,
    repeat: int = 100,
    ram_monitor_interval: float = 0.01,
) -> Dict:
    model.to(device)
    model.eval()

    gc.collect()

    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    ram_monitor = PeakRAMMonitor(interval=ram_monitor_interval)

    tracemalloc.start()

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    ram_monitor.start()

    with torch.no_grad():
        x = x.to(device)

        for _ in range(warmup):
            _ = safe_forward(model, x)

        if device.startswith("cuda"):
            torch.cuda.synchronize()

        for _ in range(repeat):
            _ = safe_forward(model, x)

        if device.startswith("cuda"):
            torch.cuda.synchronize()

    ram_peak_mb = ram_monitor.stop()

    heap_current, heap_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    heap_current_mb = heap_current / 1024 / 1024
    heap_peak_mb = heap_peak / 1024 / 1024

    if PSUTIL_AVAILABLE:
        ram_current_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    else:
        ram_current_mb = ""

    if device.startswith("cuda"):
        gpu_current_allocated_mb = torch.cuda.memory_allocated(device) / 1024 / 1024
        gpu_peak_allocated_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024
        gpu_current_reserved_mb = torch.cuda.memory_reserved(device) / 1024 / 1024
        gpu_peak_reserved_mb = torch.cuda.max_memory_reserved(device) / 1024 / 1024
    else:
        gpu_current_allocated_mb = ""
        gpu_peak_allocated_mb = ""
        gpu_current_reserved_mb = ""
        gpu_peak_reserved_mb = ""

    return {
        "ram_current_mb": ram_current_mb,
        "ram_peak_mb": ram_peak_mb,
        "heap_current_mb": heap_current_mb,
        "heap_peak_mb": heap_peak_mb,
        "gpu_current_allocated_mb": gpu_current_allocated_mb,
        "gpu_peak_allocated_mb": gpu_peak_allocated_mb,
        "gpu_current_reserved_mb": gpu_current_reserved_mb,
        "gpu_peak_reserved_mb": gpu_peak_reserved_mb,
    }


def get_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    if device_arg == "cuda":
        if torch.cuda.is_available():
            return "cuda:0"

        print("[warn] CUDA requested but not available. fallback to CPU.")
        return "cpu"

    return "cpu"


def append_csv(csv_path: str, row: Dict):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(row.keys())

    if csv_path.exists():
        with csv_path.open("r", newline="") as f:
            reader = csv.reader(f)
            existing_header = next(reader, None)

        if existing_header != fieldnames:
            backup_path = csv_path.with_suffix(".csv.bak")
            print(f"[warn] CSV header mismatch. Backup old csv to: {backup_path}")
            shutil.move(str(csv_path), str(backup_path))

    file_exists = csv_path.exists()

    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def main():
    args = get_args()

    if not PSUTIL_AVAILABLE:
        print("[warn] psutil is not installed. ram_current_mb and ram_peak_mb will be empty.")
        print("[warn] install with: pip install psutil")

    device = get_device(args.device)
    model_key = make_model_key(args)

    model = build_model(args)
    model.eval()

    x = torch.randn(
        args.batch_size,
        args.channels,
        args.signal_length
    )

    params_trainable = count_trainable_params(model)
    params_total = count_total_params(model)
    size_mb = estimate_model_size_mb(model)

    flops, thop_params = measure_flops(model, x)

    batch_latency, sample_latency, throughput = measure_latency(
        model=model,
        x=x,
        device=device,
        warmup=args.warmup,
        repeat=args.repeat,
    )

    if args.skip_memory:
        memory_stats = {
            "ram_current_mb": "",
            "ram_peak_mb": "",
            "heap_current_mb": "",
            "heap_peak_mb": "",
            "gpu_current_allocated_mb": "",
            "gpu_peak_allocated_mb": "",
            "gpu_current_reserved_mb": "",
            "gpu_peak_reserved_mb": "",
        }
    else:
        memory_repeat = args.memory_repeat if args.memory_repeat is not None else args.repeat

        memory_stats = measure_inference_memory(
            model=model,
            x=x,
            device=device,
            warmup=args.memory_warmup,
            repeat=memory_repeat,
            ram_monitor_interval=args.ram_monitor_interval,
        )

    row = {
        "dataset": args.dataset,
        "model_key": model_key,
        "model": args.model,
        "patch_size": args.patch_size,
        "reservoir_size": args.reservoir_size,
        "reservoir_rank": args.reservoir_rank if args.model in [
            "PRC_LRGR",
            "APS_LRGR",
            "IFSA_LRGR",
            "SIR_LRGR",
        ] else "",
        
        "patch_keep_ratio": args.patch_keep_ratio if args.model == "APS_LRGR" else "",
        "input_rank": args.input_rank if args.model == "IFSA_LRGR" else "",
        
        "innovation_target_ratio": (
            args.innovation_target_ratio
            if args.model == "SIR_LRGR"
            else ""
        ),
        "innovation_threshold": (
            args.innovation_threshold
            if args.model == "SIR_LRGR"
            else ""
        ),
        "innovation_hidden_dim": (
            args.innovation_hidden_dim
            if args.model == "SIR_LRGR"
            else ""
        ),
        "innovation_min_keep": (
            args.innovation_min_keep
            if args.model == "SIR_LRGR"
            else ""
        ),
        "input_shape": str(tuple(x.shape)),
        "batch_size": args.batch_size,
        "device": device,
        "params": params_total,
        "params_total": params_total,
        "params_trainable": params_trainable,
        "size_mb": size_mb,
        "flops": flops if flops is not None else "",
        "flops_m": flops / 1e6 if flops is not None else "",
        "thop_params": thop_params if thop_params is not None else "",
        "latency_batch_ms": batch_latency * 1000,
        "latency_sample_ms": sample_latency * 1000,
        "throughput_samples_per_sec": throughput,
        "ram_current_mb": memory_stats["ram_current_mb"],
        "ram_peak_mb": memory_stats["ram_peak_mb"],
        "heap_current_mb": memory_stats["heap_current_mb"],
        "heap_peak_mb": memory_stats["heap_peak_mb"],
        "gpu_current_allocated_mb": memory_stats["gpu_current_allocated_mb"],
        "gpu_peak_allocated_mb": memory_stats["gpu_peak_allocated_mb"],
        "gpu_current_reserved_mb": memory_stats["gpu_current_reserved_mb"],
        "gpu_peak_reserved_mb": memory_stats["gpu_peak_reserved_mb"],
    }

    print("========== Footprint ==========")

    for key, value in row.items():
        print(f"{key}: {value}")

    append_csv(args.csv_path, row)


if __name__ == "__main__":
    main()