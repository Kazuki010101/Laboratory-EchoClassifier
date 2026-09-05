#!/usr/bin/env python3
"""Paper-oriented SENvT layer diagnostics using the exact TGEC evidence route."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.senvt_adapters import SenvtTeacherAdapter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--acc", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split-indices", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--samples", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--input-size", type=int, default=496)
    p.add_argument("--student-patch-size", type=int, default=16)
    p.add_argument("--keep-ratio", type=float, default=0.5)
    p.add_argument("--selected-layer", type=int, default=1)
    return p.parse_args()


def load_validation_indices(path: str) -> np.ndarray:
    split = np.load(path)
    for key in ("validation_indices", "val_indices"):
        if key in split.files:
            return np.asarray(split[key], dtype=np.int64)
    raise KeyError(
        f"{path} must contain validation_indices or val_indices; found {split.files}"
    )


def stratified_sample(indices, labels, count, seed):
    rng = np.random.default_rng(seed)
    chosen = []
    classes = np.unique(labels[indices])
    base, remainder = divmod(min(count, len(indices)), len(classes))
    for order, cls in enumerate(classes):
        candidates = indices[labels[indices] == cls].copy()
        rng.shuffle(candidates)
        chosen.extend(candidates[: base + (order < remainder)].tolist())
    chosen = np.asarray(chosen, dtype=np.int64)
    rng.shuffle(chosen)
    return chosen


class LayerEvidenceCollector:
    def __init__(self, teacher, student_patch_size, keep_ratio):
        self.teacher = teacher
        self.patch_size = student_patch_size
        self.keep_ratio = keep_ratio
        self.batch_records = [[] for _ in teacher.layers]
        self.route_batches = [[] for _ in teacher.layers]
        self.handles = []
        for index, layer in enumerate(teacher.layers):
            self.handles.append(
                layer.attention.register_forward_hook(self._hook(index, layer))
            )

    def _hook(self, layer_index, layer):
        def collect(_module, inputs, output):
            token_input = inputs[0].detach()
            attention_output = output[0].detach()
            weights = output[1].detach()
            batch, tokens, dim = token_input.shape
            heads = layer.attention.nheads

            qkv = layer.attention.qkv(token_input).reshape(
                batch, tokens, 3, heads, dim // heads
            )
            value = qkv[:, :, 2].permute(0, 2, 1, 3)
            cls_attention = weights[:, :, 0, 1:]
            weighted = cls_attention.unsqueeze(-1) * value[:, :, 1:, :]
            concatenated = weighted.permute(0, 2, 1, 3).reshape(
                batch, tokens - 1, dim
            )
            time_evidence = F.linear(concatenated, layer.attention.proj.weight)
            length = time_evidence.shape[1]
            if length % self.patch_size:
                raise ValueError(
                    f"Layer {layer_index}: teacher length {length} is not divisible "
                    f"by student patch size {self.patch_size}"
                )
            patch_count = length // self.patch_size
            keep_count = max(
                1, min(patch_count, math.ceil(patch_count * self.keep_ratio))
            )
            patch_evidence = time_evidence.reshape(
                batch, patch_count, self.patch_size, dim
            ).sum(dim=2)
            route = torch.softmax(
                torch.log(patch_evidence.norm(dim=-1).clamp_min(1e-12)), dim=-1
            ).float()
            entropy = -(route * route.clamp_min(1e-12).log()).sum(-1)
            entropy = entropy / math.log(patch_count)
            topk_mass = route.topk(keep_count, dim=-1).values.sum(-1)

            input_dispersion = token_input[:, 1:].float().std(
                dim=1, unbiased=False
            ).mean(-1)
            output_dispersion = attention_output[:, 1:].float().std(
                dim=1, unbiased=False
            ).mean(-1)
            self.route_batches[layer_index].append(route.cpu().numpy())
            self.batch_records[layer_index].append({
                "entropy": entropy.cpu().numpy(),
                "topk_mass": topk_mass.cpu().numpy(),
                "route_std": route.std(dim=-1, unbiased=False).cpu().numpy(),
                "input_dispersion": input_dispersion.cpu().numpy(),
                "output_dispersion": output_dispersion.cpu().numpy(),
            })
        return collect

    def close(self):
        for handle in self.handles:
            handle.remove()

    def arrays(self):
        routes = np.stack(
            [np.concatenate(parts, axis=0) for parts in self.route_batches], axis=1
        )  # [sample, layer, patch]
        metrics = {}
        for key in self.batch_records[0][0]:
            metrics[key] = np.stack([
                np.concatenate([part[key] for part in layer_parts], axis=0)
                for layer_parts in self.batch_records
            ], axis=1)  # [sample, layer]
        return routes, metrics


def confidence_interval(values):
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    half = 1.96 * std / math.sqrt(max(1, len(values)))
    return mean, std, mean - half, mean + half


def write_csv(path, metrics, random_mass):
    fields = [
        "layer", "evidence_topk_mass_mean", "evidence_topk_mass_std",
        "evidence_topk_mass_ci95_low", "evidence_topk_mass_ci95_high",
        "oracle_advantage_over_random", "evidence_entropy_mean",
        "evidence_route_std_mean", "attention_input_dispersion_mean",
        "attention_output_dispersion_mean",
    ]
    rows = []
    for layer in range(metrics["topk_mass"].shape[1]):
        mean, std, low, high = confidence_interval(metrics["topk_mass"][:, layer])
        rows.append({
            "layer": layer,
            "evidence_topk_mass_mean": mean,
            "evidence_topk_mass_std": std,
            "evidence_topk_mass_ci95_low": low,
            "evidence_topk_mass_ci95_high": high,
            "oracle_advantage_over_random": mean - random_mass,
            "evidence_entropy_mean": float(metrics["entropy"][:, layer].mean()),
            "evidence_route_std_mean": float(metrics["route_std"][:, layer].mean()),
            "attention_input_dispersion_mean": float(metrics["input_dispersion"][:, layer].mean()),
            "attention_output_dispersion_mean": float(metrics["output_dispersion"][:, layer].mean()),
        })
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def make_figures(out, routes, metrics, selected_layer, random_mass):
    layers = np.arange(routes.shape[1])
    topk = metrics["topk_mass"]
    means = topk.mean(0)
    ci = 1.96 * topk.std(0, ddof=1) / math.sqrt(topk.shape[0])

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    axes[0].errorbar(layers, means, yerr=ci, marker="o", capsize=3,
                     color="#d64545", label="Evidence Top-16 mass (95% CI)")
    axes[0].axhline(random_mass, color="black", linestyle="--",
                    label=f"Uniform baseline = {random_mass:.3f}")
    axes[0].axvline(selected_layer, color="#f39c12", linestyle=":",
                    label=f"TGEC uses layer {selected_layer}")
    axes[0].set_ylabel("Top-16 mass")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=.25)

    axes[1].plot(layers, metrics["entropy"].mean(0), marker="o", color="#2f80c9")
    axes[1].axhline(1.0, color="black", linestyle="--")
    axes[1].axvline(selected_layer, color="#f39c12", linestyle=":")
    axes[1].set_ylabel("Normalized entropy\n(lower = more selective)")
    axes[1].grid(alpha=.25)

    axes[2].semilogy(layers, np.maximum(metrics["input_dispersion"].mean(0), 1e-12),
                     marker="o", label="Attention input")
    axes[2].semilogy(layers, np.maximum(metrics["output_dispersion"].mean(0), 1e-12),
                     marker="s", label="Attention output")
    axes[2].axvline(selected_layer, color="#f39c12", linestyle=":")
    axes[2].set_xlabel("SENvT encoder layer (zero-based)")
    axes[2].set_ylabel("Temporal-token dispersion")
    axes[2].legend(fontsize=9)
    axes[2].grid(alpha=.25)
    fig.suptitle("Layer selection using the exact TGEC Attention×Value evidence")
    fig.tight_layout()
    fig.savefig(out / "01_layer_selection.png", dpi=240, bbox_inches="tight")
    fig.savefig(out / "01_layer_selection.pdf", bbox_inches="tight")
    plt.close(fig)

    mean_route = routes.mean(0)
    relative = mean_route / (1.0 / routes.shape[2])
    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(relative, aspect="auto", cmap="RdBu_r", vmin=.75, vmax=1.25)
    ax.axhline(selected_layer, color="#f39c12", linewidth=2)
    ax.set_xlabel("Student temporal patch index")
    ax.set_ylabel("SENvT encoder layer")
    ax.set_yticks(layers)
    ax.set_title("Mean TGEC evidence route relative to a uniform route")
    fig.colorbar(im, ax=ax, label="q / (1/31)")
    fig.tight_layout()
    fig.savefig(out / "02_layer_patch_heatmap.png", dpi=240, bbox_inches="tight")
    fig.savefig(out / "02_layer_patch_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)

    advantage = metrics["topk_mass"] - random_mass
    order = np.argsort(advantage[:, selected_layer])[::-1]
    shown = order[:min(128, len(order))]
    fig, ax = plt.subplots(figsize=(10, 7))
    limit = max(.01, float(np.abs(advantage[shown]).max()))
    im = ax.imshow(advantage[shown], aspect="auto", cmap="RdBu_r",
                   vmin=-limit, vmax=limit)
    ax.axvline(selected_layer, color="#f39c12", linewidth=2)
    ax.set_xlabel("SENvT encoder layer")
    ax.set_ylabel("Validation samples (sorted by layer-1 advantage)")
    ax.set_title("Per-sample Top-16 evidence advantage over uniform selection")
    fig.colorbar(im, ax=ax, label="Top-16 mass − 16/31")
    fig.tight_layout()
    fig.savefig(out / "03_sample_layer_heatmap.png", dpi=240, bbox_inches="tight")
    fig.savefig(out / "03_sample_layer_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    if not 0 < args.keep_ratio <= 1:
        raise ValueError("keep_ratio must be in (0, 1]")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    acc = np.load(args.acc, mmap_mode="r")
    labels = np.asarray(np.load(args.label, mmap_mode="r"), dtype=np.int64) - 1
    validation = load_validation_indices(args.split_indices)
    sampled = stratified_sample(validation, labels, args.samples, args.seed)
    if len(sampled) == 0:
        raise RuntimeError("No validation samples were selected")

    device = torch.device(args.device)
    teacher = SenvtTeacherAdapter(
        variant="B", num_classes=8, window_size=args.input_size, in_chans=3,
        ckpt_path=args.checkpoint, device=device,
    )
    core = teacher.core
    collector = LayerEvidenceCollector(
        core, args.student_patch_size, args.keep_ratio
    )
    correct = 0
    try:
        with torch.inference_mode():
            for start in range(0, len(sampled), args.batch_size):
                ids = sampled[start:start + args.batch_size]
                x = torch.from_numpy(
                    np.array(acc[ids], dtype=np.float32, copy=True)
                ).to(device)
                if x.shape[-1] != args.input_size:
                    x = F.interpolate(x, size=args.input_size, mode="linear",
                                      align_corners=False)
                logits = core(x)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                y = torch.from_numpy(labels[ids]).to(device)
                correct += int((logits.argmax(-1) == y).sum().item())
                print(f"batch={start // args.batch_size + 1}/"
                      f"{math.ceil(len(sampled) / args.batch_size)}")
    finally:
        collector.close()

    routes, metrics = collector.arrays()
    patch_count = routes.shape[-1]
    keep_count = math.ceil(patch_count * args.keep_ratio)
    random_mass = keep_count / patch_count
    rows = write_csv(out / "senvt_layer_metrics.csv", metrics, random_mass)
    make_figures(out, routes, metrics, args.selected_layer, random_mass)
    best_layer = int(np.argmax(metrics["topk_mass"].mean(0)))

    np.savez_compressed(
        out / "senvt_layer_sample_metrics.npz",
        source_indices=sampled,
        routes=routes.astype(np.float16),
        topk_mass=metrics["topk_mass"].astype(np.float32),
        entropy=metrics["entropy"].astype(np.float32),
        route_std=metrics["route_std"].astype(np.float32),
        input_dispersion=metrics["input_dispersion"].astype(np.float32),
        output_dispersion=metrics["output_dispersion"].astype(np.float32),
    )
    report = {
        "method": "Exact TGEC CLS Attention×Value patch-evidence norm",
        "samples": int(len(sampled)),
        "sampling": "class-stratified subset of fixed User1 validation indices",
        "teacher_accuracy_on_sample": correct / len(sampled),
        "patch_count": patch_count,
        "keep_count": keep_count,
        "uniform_topk_mass": random_mass,
        "configured_tgec_layer": args.selected_layer,
        "empirical_best_layer_by_mean_topk_mass": best_layer,
        "configured_layer_matches_empirical_best": best_layer == args.selected_layer,
        "layers": rows,
        "settings": vars(args),
    }
    (out / "senvt_layer_diagnostics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "samples": len(sampled), "teacher_accuracy": correct / len(sampled),
        "configured_layer": args.selected_layer, "empirical_best_layer": best_layer,
        "output_dir": str(out),
    }, indent=2))


if __name__ == "__main__":
    main()
