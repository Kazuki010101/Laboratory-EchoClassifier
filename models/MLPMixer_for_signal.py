import torch.nn as nn
from timm.layers import to_2tuple

class PatchEmbed1D(nn.Module):
    """ 1D Signal to Patch Embedding
    """
    def __init__(
            self,
            signal_len=1024,
            patch_size=64,
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        self.signal_len = signal_len
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.num_patches = signal_len // patch_size
        self.flatten = flatten

        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, L = x.shape
        assert L == self.signal_len, "Input signal length ({L}) doesn't match model ({self.signal_len})."
        x = self.proj(x)
        x = self.norm(x)
        return x

class MlpBlock(nn.Module):
    def __init__(self, hidden_dim, mlp_dim):
        super(MlpBlock, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_dim)
        )

    def forward(self, x):
        return self.mlp(x)

class MixerBlock(nn.Module):
    def __init__(self, num_tokens, hidden_dim, tokens_mlp_dim, channels_mlp_dim):
        super(MixerBlock, self).__init__()
        self.ln_token = nn.LayerNorm(hidden_dim)
        self.token_mix = MlpBlock(num_tokens, tokens_mlp_dim)
        self.ln_channel = nn.LayerNorm(hidden_dim)
        self.channel_mix = MlpBlock(hidden_dim, channels_mlp_dim)

    def forward(self, x):
        # Token mixing
        out = self.ln_token(x)
        out = out.transpose(1, 2)  # (batch_size, sequence_length, channels)
        out = self.token_mix(out)
        out = out.transpose(1, 2)  # (batch_size, channels, sequence_length)
        out = x + out  # Residual connection

        # Channel mixing
        out = self.ln_channel(out)
        out = self.channel_mix(out)
        out = out + x  # Residual connection

        return out

class MLPMixer(nn.Module):
    def __init__(self, in_channels, dim, num_classes, patch_size, sequence_length, depth, mlp_ratio=(0.5, 4.0)):
        super().__init__()
        assert sequence_length % patch_size == 0, 'Sequence length must be divisible by the patch size.'
        token_dim, channel_dim = [int(x * dim) for x in to_2tuple(mlp_ratio)]
        self.num_patch = (sequence_length // patch_size)
        self.to_patch_embedding = PatchEmbed1D(signal_len=sequence_length, patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=dim,
            norm_layer=None,
            flatten=True,
            bias=True,)
        self.mixer_blocks = nn.ModuleList([])
        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock(dim, self.num_patch, token_dim, channel_dim))
        self.layer_norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.to_patch_embedding(x)
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        x = x.transpose(1,2)
        x = self.layer_norm(x)
        x = x.mean(dim=1)
        return self.head(x)