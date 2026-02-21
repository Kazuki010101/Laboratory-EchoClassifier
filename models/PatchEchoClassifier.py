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

class ReservoirNetwork(nn.Module):
    def __init__(self, input_size, reservoir_size, spectral_radius=0.9):
        super(ReservoirNetwork, self).__init__()
        self.reservoir_size = reservoir_size
        self.W_reservoir = nn.Parameter(torch.rand(reservoir_size, reservoir_size) - 0.5, requires_grad=False)
        self.W_input = nn.Parameter(torch.rand(reservoir_size, input_size) - 0.5, requires_grad=False)
        
        # Scale the reservoir weights to ensure echo state property
        eigenvalues = torch.linalg.eigvals(self.W_reservoir)
        max_eigenvalue = torch.max(torch.abs(eigenvalues))
        self.W_reservoir.data *= spectral_radius / max_eigenvalue

    def forward(self, x, cls, dist):
        batch_size, _, seq_length = x.size()
        h = torch.zeros(batch_size, self.reservoir_size, device=x.device)
        
        for t in range(seq_length):
            u = x[:, :, t]
            h = torch.tanh(torch.matmul(h, self.W_reservoir) + torch.matmul(u, self.W_input.T))
            
        h_cls = torch.tanh(torch.matmul(h, self.W_reservoir) + torch.matmul(cls, self.W_input.T))
        h_dist = torch.tanh(torch.matmul(h, self.W_reservoir) + torch.matmul(dist, self.W_input.T))
        
        return h_cls, h_dist

class ClassificationHead(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.fc(x)

class PatchReservoir(nn.Module):
    def __init__(self, in_channels, patch_size, stride, reservoir_size, num_classes):
        super(PatchReservoir, self).__init__()
        self.patch_embed = PatchEmbed(in_channels, patch_size, stride)
        self.reservoir_network = ReservoirNetwork(patch_size, reservoir_size)
        
        # Add class and distillation tokens
        self.cls_token = nn.Parameter(torch.zeros(1, patch_size))
        self.dist_token = nn.Parameter(torch.zeros(1, patch_size))
        
        self.classification_head = ClassificationHead(reservoir_size, num_classes)
        self.distillation_head = ClassificationHead(reservoir_size, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        
        # Add class and distillation tokens to the sequence
        cls_tokens = self.cls_token.expand(x.shape[0], -1)
        dist_tokens = self.dist_token.expand(x.shape[0], -1)
        
        x_cls, x_dist = self.reservoir_network(x, cls_tokens, dist_tokens)
        
        # Separate the outputs for class and distillation tokens
        cls_output = self.classification_head(x_cls)
        dist_output = self.distillation_head(x_dist)
        
        if self.training:
            return F.log_softmax(cls_output, dim=1), F.log_softmax(dist_output, dim=1)
        else:
            # During inference, return the average of both classifier predictions
            return F.log_softmax((cls_output + dist_output) / 2, dim=1)
