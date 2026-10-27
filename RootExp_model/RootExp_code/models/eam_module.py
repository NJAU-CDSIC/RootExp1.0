"""
Edge Attention Module (EAM) – same as U-Mamba_EAM_adapted (InstanceNorm2d + LeakyReLU).

Extracts edge-aware features from the original input image via dilated
convolutions and CBAM attention; the network fuses them with bottleneck
features before SC_Att_Bridge.
"""
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        hidden = max(1, in_planes // ratio)
        self.fc1 = nn.Conv2d(in_planes, hidden, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(hidden, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


class EAM(nn.Module):
    """Edge Attention Module using InstanceNorm2d to match U-Mamba's norm."""

    def __init__(self, in_channels=3, out_channels=1024):
        super().__init__()
        self.eam = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(64, eps=1e-5, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.InstanceNorm2d(128, eps=1e-5, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.InstanceNorm2d(256, eps=1e-5, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, out_channels, kernel_size=3, stride=1, padding=5, dilation=5),
            nn.InstanceNorm2d(out_channels, eps=1e-5, affine=True),
            nn.LeakyReLU(inplace=True),
            CBAM(in_planes=out_channels, ratio=16, kernel_size=7),
        )

    def forward(self, x):
        return self.eam(x)
