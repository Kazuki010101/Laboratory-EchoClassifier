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


class InputFactorizedStateAttentiveLowRankGatedReservoirNetwork(nn.Module):
    def __init__(
        self,
        input_size,
        reservoir_size,
        reservoir_rank=64,
        input_rank=16,
        leak_rate=1.0,
        attention_hidden=None,
        use_layer_norm=True,
    ):
        super().__init__()

        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.reservoir_rank = reservoir_rank
        self.input_rank = input_rank
        self.leak_rate = leak_rate

        self.U = nn.Parameter(
            torch.randn(reservoir_size, reservoir_rank) * 0.02,
            requires_grad=False
        )

        self.V = nn.Parameter(
            torch.randn(reservoir_rank, reservoir_size) * 0.02,
            requires_grad=False
        )

        self.input_down = nn.Parameter(
            torch.randn(input_rank, input_size) * 0.02,
            requires_grad=False
        )

        self.input_up = nn.Parameter(
            torch.randn(reservoir_size, input_rank) * 0.02,
            requires_grad=False
        )

        self.gate = nn.Linear(input_rank, 1)

        if attention_hidden is None:
            attention_hidden = max(16, min(128, reservoir_size // 4))

        self.state_attention = nn.Sequential(
            nn.Linear(reservoir_size, attention_hidden),
            nn.Tanh(),
            nn.Linear(attention_hidden, 1)
        )

        if use_layer_norm:
            self.norm = nn.LayerNorm(reservoir_size)
        else:
            self.norm = nn.Identity()

    def recurrent(self, h):
        return torch.matmul(torch.matmul(h, self.U), self.V)

    def input_factorized_drive(self, u):
        q = torch.matmul(u, self.input_down.T)
        drive = torch.matmul(q, self.input_up.T)
        return q, drive

    def attentive_pooling(self, states):
        scores = self.state_attention(states).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(states * weights.unsqueeze(-1), dim=1)
        return pooled, weights

    def forward(self, x, cls, dist):
        batch_size, _, seq_length = x.size()

        h = torch.zeros(
            batch_size,
            self.reservoir_size,
            device=x.device,
            dtype=x.dtype
        )

        states = []

        for t in range(seq_length):
            u = x[:, :, t]

            q, input_drive = self.input_factorized_drive(u)

            h_new = torch.tanh(
                self.recurrent(h) + input_drive
            )

            g = torch.sigmoid(self.gate(q))
            effective_gate = self.leak_rate * g

            h = (1.0 - effective_gate) * h + effective_gate * h_new
            h = self.norm(h)

            states.append(h.unsqueeze(1))

        states = torch.cat(states, dim=1)
        h_pool, attention_weights = self.attentive_pooling(states)

        _, cls_drive = self.input_factorized_drive(cls)
        _, dist_drive = self.input_factorized_drive(dist)

        h_cls_new = torch.tanh(
            self.recurrent(h) + cls_drive
        )

        h_dist_new = torch.tanh(
            self.recurrent(h) + dist_drive
        )

        h_cls = self.norm(h_cls_new + h_pool)
        h_dist = self.norm(h_dist_new + h_pool)

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
        input_rank,
        num_classes,
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            patch_size=patch_size,
            stride=stride
        )

        self.reservoir_network = InputFactorizedStateAttentiveLowRankGatedReservoirNetwork(
            input_size=patch_size,
            reservoir_size=reservoir_size,
            reservoir_rank=reservoir_rank,
            input_rank=input_rank,
            leak_rate=1.0,
            attention_hidden=None,
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

        return (cls_output + dist_output) / 2