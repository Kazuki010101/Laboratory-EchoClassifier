import math
import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, in_channels, patch_size, stride):
        super().__init__()
        self.conv1d = nn.Conv1d(
            in_channels,
            patch_size,
            kernel_size=patch_size,
            stride=stride
        )

    def forward(self, x):
        return self.conv1d(x)


class AdaptivePatchSkippingLowRankGatedReservoirNetwork(nn.Module):
    def __init__(
        self,
        input_size,
        reservoir_size,
        rank=64,
        keep_ratio=0.5,
        leak_rate=0.5,
        use_layer_norm=True,
    ):
        super().__init__()

        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.rank = rank
        self.keep_ratio = keep_ratio
        self.leak_rate = leak_rate

        # Low-rank reservoir: W_reservoir ≈ U @ V
        self.U = nn.Parameter(
            torch.randn(reservoir_size, rank) * 0.02,
            requires_grad=False
        )
        self.V = nn.Parameter(
            torch.randn(rank, reservoir_size) * 0.02,
            requires_grad=False
        )

        self.W_input = nn.Parameter(
            torch.randn(reservoir_size, input_size) * 0.02,
            requires_grad=False
        )

        # update gate: selected patchごとの状態更新量を調整
        self.gate = nn.Linear(input_size, 1)

        # patch importance scorer: どのpatchを使うか決める
        self.patch_scorer = nn.Linear(input_size, 1)

        if use_layer_norm:
            self.norm = nn.LayerNorm(reservoir_size)
        else:
            self.norm = nn.Identity()

    def recurrent(self, h):
        return torch.matmul(torch.matmul(h, self.U), self.V)

    def select_patches(self, x):
        """
        x: [B, D, T]
        return:
            selected_patches: [B, K, D]
            selected_scores:  [B, K]
        """
        patches = x.transpose(1, 2)  # [B, T, D]
        batch_size, seq_len, dim = patches.shape

        scores = torch.sigmoid(self.patch_scorer(patches)).squeeze(-1)  # [B, T]

        keep_k = int(math.ceil(seq_len * self.keep_ratio))
        keep_k = max(1, min(seq_len, keep_k))

        if keep_k < seq_len:
            topk_idx = torch.topk(scores, k=keep_k, dim=1).indices  # [B, K]

            # 時系列順を保つ
            selected_idx = torch.sort(topk_idx, dim=1).values
        else:
            selected_idx = torch.arange(seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)

        gather_idx = selected_idx.unsqueeze(-1).expand(-1, -1, dim)
        selected_patches = torch.gather(patches, dim=1, index=gather_idx)
        selected_scores = torch.gather(scores, dim=1, index=selected_idx)

        return selected_patches, selected_scores

    def forward(self, x, cls, dist):
        """
        x: [B, D, T]
        cls, dist: [B, D]
        """
        batch_size, _, _ = x.size()

        selected_patches, selected_scores = self.select_patches(x)
        keep_k = selected_patches.size(1)

        h = torch.zeros(
            batch_size,
            self.reservoir_size,
            device=x.device
        )

        h_sum = torch.zeros_like(h)

        for t in range(keep_k):
            u = selected_patches[:, t, :]          # [B, D]
            importance = selected_scores[:, t:t+1] # [B, 1]

            h_new = torch.tanh(
                self.recurrent(h) + torch.matmul(u, self.W_input.T)
            )

            update_gate = torch.sigmoid(self.gate(u))

            # patch importance と update gate を合成
            g = update_gate * importance

            h = (1.0 - g) * h + g * h_new
            h = self.norm(h)

            h_sum = h_sum + h

        h_mean = h_sum / keep_k

        h_cls_new = torch.tanh(
            self.recurrent(h) + torch.matmul(cls, self.W_input.T)
        )

        h_dist_new = torch.tanh(
            self.recurrent(h) + torch.matmul(dist, self.W_input.T)
        )

        h_cls = self.norm(h_cls_new + h_mean)
        h_dist = self.norm(h_dist_new + h_mean)

        return h_cls, h_dist


class ClassificationHead(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.fc(x)


class PatchReservoir(nn.Module):
    def __init__(
        self,
        in_channels,
        patch_size,
        stride,
        reservoir_size,
        reservoir_rank,
        patch_keep_ratio,
        num_classes,
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            patch_size=patch_size,
            stride=stride
        )

        self.reservoir_network = AdaptivePatchSkippingLowRankGatedReservoirNetwork(
            input_size=patch_size,
            reservoir_size=reservoir_size,
            rank=reservoir_rank,
            keep_ratio=patch_keep_ratio,
            leak_rate=0.5,
            use_layer_norm=True,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, patch_size))
        self.dist_token = nn.Parameter(torch.zeros(1, patch_size))

        self.classification_head = ClassificationHead(
            reservoir_size,
            num_classes
        )

        self.distillation_head = ClassificationHead(
            reservoir_size,
            num_classes
        )

    def forward(self, x):
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(x.shape[0], -1)
        dist_tokens = self.dist_token.expand(x.shape[0], -1)

        x_cls, x_dist = self.reservoir_network(
            x,
            cls_tokens,
            dist_tokens
        )

        cls_output = self.classification_head(x_cls)
        dist_output = self.distillation_head(x_dist)

        if self.training:
            return cls_output, dist_output
        else:
            return (cls_output + dist_output) / 2