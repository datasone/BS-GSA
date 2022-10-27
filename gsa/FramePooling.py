import torch
from torch import nn
from torch.nn import functional as F


class ChannelZeroPadding(nn.Module):
    def __init__(self, target_channel: int):
        super().__init__()
        self.target_channel = target_channel

    def forward(self, x):
        b, c = x.size()
        padding_channels = self.target_channel - c
        assert padding_channels > 0, "Input channels larger than specified"
        return torch.cat([x, torch.zeros((b, padding_channels), device='cuda')], dim=1)


class FramePooling(nn.Module):
    def __init__(self, linear_dim: int, target_channel: int):
        super().__init__()
        self.linear_dim = linear_dim
        self.gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.fc = nn.Sequential(
            ChannelZeroPadding(target_channel),
            nn.Linear(target_channel, linear_dim),
            nn.ReLU(),
            nn.Linear(linear_dim, target_channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.gap(x).view(b, c)
        y = self.fc(y)
        y = F.adaptive_avg_pool1d(y, c).view(b, c, 1, 1)
        return x * y.expand_as(x)
