import torch.nn as nn
import torch
from timm.layers import to_2tuple
from timm.models.layers import trunc_normal_

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
        out = out.transpose(1, 2)  # (batch_size, hidden_dim, num_tokens)
        out = self.token_mix(out)
        out = out.transpose(1, 2)  # (batch_size, num_tokens, hidden_dim)
        x = x + out  # Residual connection

        # Channel mixing
        out = self.ln_channel(x)
        out = self.channel_mix(out)
        x = x + out  # Residual connection

        return x

class DistilledMLPMixer(nn.Module):
    def __init__(self, in_channels=3, dim=1024, num_classes=8, patch_size=16, sequence_length=496, depth=12, mlp_ratio=(0.5, 4.0)):
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
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patch + 2, dim))
        
        self.mixer_blocks = nn.ModuleList([])
        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock(dim, self.num_patch + 2, token_dim, channel_dim))
        
        self.layer_norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self.head_dist = nn.Linear(dim, num_classes)
        
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.to_patch_embedding(x)
        
        x = x.transpose(1,2)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        dist_tokens = self.dist_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, dist_tokens, x), dim=1)
        
        x = x + self.pos_embed
        
        x = x.transpose(1,2)
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        
        x = x.transpose(1,2)
        x = self.layer_norm(x)
        
        cls_token, dist_token = x[:, 0], x[:, 1]
        
        x_cls = self.head(cls_token)
        x_dist = self.head_dist(dist_token)
        
        if self.training:
            return x_cls, x_dist
        else:
            return (x_cls + x_dist) / 2