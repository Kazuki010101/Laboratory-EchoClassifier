import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    def __init__(self, in_channels, patch_size, stride):
        super(PatchEmbed, self).__init__()
        self.conv1d = nn.Conv1d(in_channels, patch_size, kernel_size=patch_size, stride=stride)

    def forward(self, x):
        # Apply 1D convolution to segment the signal into patches
        return self.conv1d(x)

# --- Self-Attention Block ---
class Attention(nn.Module):
    def __init__(self, query_dim, key_dim, num_heads=1):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        
        # Q, K, V の出力次元を、クエリ次元に合わせる
        head_dim = query_dim // num_heads
        self.scale = head_dim ** -0.5

        # クエリは h から生成されるため、入力次元は query_dim (reservoir_size)
        self.q = nn.Linear(query_dim, query_dim)
        # キーとバリューは x から生成されるため、入力次元は key_dim (patch_size)
        self.k = nn.Linear(key_dim, query_dim)
        self.v = nn.Linear(key_dim, query_dim)

        self.proj = nn.Linear(query_dim, query_dim)

    def forward(self, x, h):
        B, N, _ = x.shape  # x.shape is (B, num_patches, patch_size)
        query_dim = self.q.out_features
        
        # Generate Q from reservoir state h
        q = self.q(h).unsqueeze(1).reshape(B, 1, self.num_heads, query_dim // self.num_heads).permute(0, 2, 1, 3)
        # Generate K, V from patch embeddings x
        k = self.k(x).reshape(B, N, self.num_heads, query_dim // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, query_dim // self.num_heads).permute(0, 2, 1, 3)
        
        # Calculate attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # Apply attention to V and reshape
        x = (attn @ v).transpose(1, 2).reshape(B, 1, query_dim)
        x = self.proj(x)
        return x.squeeze(1)
    
# --- Updated Reservoir Network with Attention ---
class ReservoirNetwork(nn.Module):
    def __init__(self, input_size, reservoir_size, spectral_radius=0.9, sparsity=0.9, num_heads=1):
        super(ReservoirNetwork, self).__init__()
        self.reservoir_size = reservoir_size
        self.attention = Attention(query_dim=reservoir_size, key_dim=input_size, num_heads=num_heads)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- 疎なリザバー行列を生成 ---
        dense_matrix = torch.rand(reservoir_size, reservoir_size, device=device) - 0.5
        mask = (torch.rand(reservoir_size, reservoir_size, device=device) > sparsity).float()  # 1-sparsity の割合だけ非ゼロ
        dense_matrix = dense_matrix * mask

        # 固有値スケーリング
        eigenvalues = torch.linalg.eigvals(dense_matrix)
        max_eigenvalue = torch.max(torch.abs(eigenvalues))
        dense_matrix *= spectral_radius / max_eigenvalue

        # 疎行列に変換
        indices = dense_matrix.nonzero().t()
        values = dense_matrix[indices[0], indices[1]]
        self.W_reservoir = nn.Parameter(
        torch.sparse_coo_tensor(indices, values, (reservoir_size, reservoir_size)).to_dense(),
        requires_grad=False)

        # 入力行列（こちらは密でOK）
        self.W_input = nn.Parameter(torch.rand(reservoir_size, input_size) - 0.5, requires_grad=False)

    def forward(self, x, cls, dist):
        batch_size, seq_length, _ = x.size()
        h = torch.zeros(batch_size, self.reservoir_size, device=x.device)

        # --- 時系列処理 ---
        for t in range(seq_length):
            u = x[:, t, :]  # (B, input_size)

            # h @ W_reservoir (疎行列積)
            h_res = torch.matmul(h, self.W_reservoir.to_dense())  # torch.sparse.mm も可
            h_in = torch.matmul(u, self.W_input.T)
            h = torch.tanh(h_res + h_in)

        # --- アテンション ---
        attended_h = self.attention(x, h)

        # --- トークン処理 ---
        h_cls = torch.tanh(torch.matmul(attended_h, self.W_reservoir.to_dense()) + torch.matmul(cls, self.W_input.T))
        h_dist = torch.tanh(torch.matmul(attended_h, self.W_reservoir.to_dense()) + torch.matmul(dist, self.W_input.T))

        return h_cls, h_dist

    
class ClassificationHead(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.fc(x)

# --- Classification and PatchEmbed remain the same ---

class PatchReservoir(nn.Module):
    def __init__(self, in_channels, patch_size, stride, reservoir_size, num_classes):
        super(PatchReservoir, self).__init__()
        self.patch_embed = PatchEmbed(in_channels, patch_size, stride)
        self.reservoir_network = ReservoirNetwork(patch_size, reservoir_size)
        
        self.cls_token = nn.Parameter(torch.zeros(1, patch_size))
        self.dist_token = nn.Parameter(torch.zeros(1, patch_size))
        
        self.classification_head = ClassificationHead(reservoir_size, num_classes)
        self.distillation_head = ClassificationHead(reservoir_size, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.transpose(1, 2)  # Change shape from (B, C, L) to (B, L, C)
        
        cls_tokens = self.cls_token.expand(x.shape[0], -1)
        dist_tokens = self.dist_token.expand(x.shape[0], -1)
        
        x_cls, x_dist = self.reservoir_network(x, cls_tokens, dist_tokens)
        
        cls_output = self.classification_head(x_cls)
        dist_output = self.distillation_head(x_dist)
        
        # if self.training:
        #     return F.log_softmax(cls_output, dim=1), F.log_softmax(dist_output, dim=1)
        # else:
        #     return F.log_softmax((cls_output + dist_output) / 2, dim=1)

        if self.training:
            return cls_output, dist_output
        else:
            return (cls_output + dist_output) / 2