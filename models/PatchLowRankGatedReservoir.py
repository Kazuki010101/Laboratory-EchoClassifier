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


class LowRankGatedReservoirNetwork(nn.Module):
    def __init__(
        self,
        input_size,
        reservoir_size,
        rank=64,
        leak_rate=0.5,
        use_layer_norm=True,
    ):
        super().__init__()

        self.reservoir_size = reservoir_size
        self.rank = rank
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

        # 入力ごとに reservoir 更新量を調整する gate
        self.gate = nn.Linear(input_size, 1)

        if use_layer_norm:
            self.norm = nn.LayerNorm(reservoir_size)
        else:
            self.norm = nn.Identity()

    def recurrent(self, h):
        return torch.matmul(torch.matmul(h, self.U), self.V)

    def forward(self, x, cls, dist):
        batch_size, _, seq_length = x.size()

        h = torch.zeros(
            batch_size,
            self.reservoir_size,
            device=x.device
        )

        h_sum = torch.zeros_like(h)

        for t in range(seq_length):
            u = x[:, :, t]

            h_new = torch.tanh(
                self.recurrent(h) + torch.matmul(u, self.W_input.T)
            )

            g = torch.sigmoid(self.gate(u))

            h = (1.0 - g) * h + g * h_new
            h = self.norm(h)

            h_sum = h_sum + h

        h_mean = h_sum / seq_length

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
        num_classes,
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            patch_size=patch_size,
            stride=stride
        )

        self.reservoir_network = LowRankGatedReservoirNetwork(
            input_size=patch_size,
            reservoir_size=reservoir_size,
            rank=reservoir_rank,
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