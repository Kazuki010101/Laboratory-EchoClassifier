"""Full-rank PRC routing and teacher-guided temporal evidence condensation."""
import math
import torch
from torch import nn


class PatchEmbed(nn.Module):
    def __init__(self, in_channels, patch_size, stride):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, patch_size, patch_size, stride=stride)

    def forward(self, x):
        return self.proj(x).transpose(1, 2)  # B,T,D


class FullRankReservoir(nn.Module):
    def __init__(self, dim, size, spectral_radius=0.9):
        super().__init__()
        w = torch.rand(size, size) - 0.5
        radius = torch.linalg.eigvals(w).abs().max().real.clamp_min(1e-8)
        self.register_buffer("W_reservoir", w * (spectral_radius / radius))
        self.register_buffer("W_input", torch.rand(size, dim) - 0.5)
        self.size = size

    def forward(self, tokens, cls_token, dist_token):
        h = tokens.new_zeros(tokens.shape[0], self.size)
        for token in tokens.unbind(dim=1):
            h = torch.tanh(h @ self.W_reservoir + token @ self.W_input.T)
        h_cls = torch.tanh(h @ self.W_reservoir + cls_token @ self.W_input.T)
        h_dist = torch.tanh(h @ self.W_reservoir + dist_token @ self.W_input.T)
        return h_cls, h_dist


def _batched_gather(tokens, indices):
    return tokens.gather(1, indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))


class PatchReservoir(nn.Module):
    """Modes: aps (ordinary), teacher_skip, and tgec (proposed)."""
    def __init__(self, in_channels, patch_size, stride, reservoir_size,
                 num_classes, patch_keep_ratio=0.5, mode="aps"):
        super().__init__()
        if mode not in {"aps", "teacher_skip", "tgec"}:
            raise ValueError(f"Unknown routing mode: {mode}")
        self.mode = mode
        self.keep_ratio = patch_keep_ratio
        self.patch_embed = PatchEmbed(in_channels, patch_size, stride)
        hidden = max(16, patch_size * 2)
        self.router = nn.Sequential(nn.LayerNorm(patch_size), nn.Linear(patch_size, hidden),
                                    nn.GELU(), nn.Linear(hidden, 1))
        self.summary_encoder = None
        if mode == "tgec":
            self.summary_encoder = nn.Sequential(
                nn.Linear(2 * patch_size, 2 * patch_size), nn.GELU(),
                nn.Linear(2 * patch_size, patch_size), nn.LayerNorm(patch_size))
        self.reservoir_network = FullRankReservoir(patch_size, reservoir_size)
        self.cls_token = nn.Parameter(torch.zeros(1, patch_size))
        self.dist_token = nn.Parameter(torch.zeros(1, patch_size))
        self.classification_head = nn.Linear(reservoir_size, num_classes)
        self.distillation_head = nn.Linear(reservoir_size, num_classes)
        self._routing_stats = {}

    def forward(self, x):
        patches = self.patch_embed(x)
        route_logits = self.router(patches).squeeze(-1)
        patch_count = patches.shape[1]
        keep_count = max(1, min(patch_count, math.ceil(patch_count * self.keep_ratio)))
        indices = route_logits.topk(keep_count, dim=1).indices.sort(dim=1).values
        selected = _batched_gather(patches, indices)

        # Ordinary APS receives task gradients through these gates. Teacher-guided
        # variants receive an explicit route-distribution loss as well.
        selected_scores = _batched_gather(route_logits.unsqueeze(-1), indices).squeeze(-1)
        selected = selected * (2.0 * torch.sigmoid(selected_scores)).unsqueeze(-1)
        selected_mask = torch.zeros_like(route_logits, dtype=torch.bool).scatter(1, indices, True)
        omitted_mask = ~selected_mask
        summary = None
        if self.mode == "tgec":
            omitted_f = omitted_mask.unsqueeze(-1).to(patches.dtype)
            count = omitted_f.sum(dim=1).clamp_min(1.0)
            mean = (patches * omitted_f).sum(dim=1) / count
            rms = ((patches.square() * omitted_f).sum(dim=1) / count).clamp_min(1e-8).sqrt()
            summary = self.summary_encoder(torch.cat([mean, rms], dim=-1))
            selected = torch.cat([selected, summary.unsqueeze(1)], dim=1)

        cls = self.cls_token.expand(x.shape[0], -1)
        dist = self.dist_token.expand(x.shape[0], -1)
        h_cls, h_dist = self.reservoir_network(selected, cls, dist)
        cls_logits = self.classification_head(h_cls)
        dist_logits = self.distillation_head(h_dist)
        with torch.no_grad():
            p = torch.softmax(route_logits, dim=1)
            ent = -(p * p.clamp_min(1e-12).log()).sum(1).mean() / math.log(patch_count)
            self._routing_stats = {"patch_count": float(patch_count), "keep_count": float(keep_count),
                                   "keep_ratio": keep_count / patch_count, "route_entropy": float(ent)}
        if self.training:
            aux = {"method": self.mode, "route_logits": route_logits,
                   "selected_indices": indices, "omitted_mask": omitted_mask,
                   "summary_token": summary, "patches": patches}
            return cls_logits, dist_logits, aux
        return (cls_logits + dist_logits) / 2

    def get_routing_stats(self):
        return self._routing_stats
