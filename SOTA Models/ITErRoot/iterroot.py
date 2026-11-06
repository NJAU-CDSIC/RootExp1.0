"""
ITErRoot - Iterative Refinement UNet for root segmentation
Adapted from: Seidenthal et al. - ITErRoot
论文架构 + GroupNorm 替代 BN、转置卷积上采样，保证收敛。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm_layer(ch):
    """GroupNorm 替代 BN，论文无 BN，用 GN 保证收敛"""
    return nn.GroupNorm(min(8, ch), ch)


class ConvBlock(nn.Module):
    """论文架构：Conv+ReLU，用 GroupNorm 替代 BN"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            _norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            _norm_layer(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), ConvBlock(in_channels, out_channels))

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    """论文：2x2 转置卷积上采样"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.block = ConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        return self.block(torch.cat([x2, x1], dim=1))


class InputBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = ConvBlock(in_channels, out_channels)
    def forward(self, x):
        return self.block(x)


class OutputBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
    def forward(self, x):
        return torch.sigmoid(self.conv(x))


class MainNet(nn.Module):
    """Larger UNet (4 down + 4 up)"""
    def __init__(self, n_channels, out_channels):
        super().__init__()
        self.input = InputBlock(n_channels, out_channels)
        self.down1 = Down(out_channels, out_channels * 2)
        self.down2 = Down(out_channels * 2, out_channels * 4)
        self.down3 = Down(out_channels * 4, out_channels * 8)
        self.down4 = Down(out_channels * 8, out_channels * 16)
        self.up1 = Up(out_channels * 16, out_channels * 8)
        self.up2 = Up(out_channels * 8, out_channels * 4)
        self.up3 = Up(out_channels * 4, out_channels * 2)
        self.up4 = Up(out_channels * 2, out_channels)
        self.output = OutputBlock(out_channels, 1)

    def forward(self, x):
        x1 = self.input(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.output(x)
        return x1, x, out


class SecondaryNet(nn.Module):
    """Smaller UNet (3 down + 3 up)，原版格式"""
    def __init__(self, n_channels, out_channels):
        super().__init__()
        self.input = InputBlock(n_channels, out_channels)
        self.down1 = Down(out_channels, out_channels * 2)
        self.down2 = Down(out_channels * 2, out_channels * 4)
        self.down3 = Down(out_channels * 4, out_channels * 8)
        self.up1 = Up(out_channels * 8, out_channels * 4)
        self.up2 = Up(out_channels * 4, out_channels * 2)
        self.up3 = Up(out_channels * 2, out_channels)
        self.output = OutputBlock(out_channels, 1)

    def forward(self, x):
        x1 = self.input(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        out = self.output(x)
        return x1, x, out


class ITErRoot(nn.Module):
    """
    ITErRoot: Iterative refinement segmentation network.
    Main UNet produces initial prediction, then secondary UNets
    iteratively refine using concatenated features.
    
    Returns only the final output (single tensor with sigmoid).
    """
    def __init__(self, num_classes=1, input_channels=3, base_channels=32, iters=2):
        super().__init__()
        self.iters = iters
        self.main_net = MainNet(input_channels, base_channels)
        self.secondary_nets = nn.ModuleList(
            SecondaryNet(base_channels * (i + 2), out_channels=base_channels)
            for i in range(iters)
        )

    def forward(self, x):
        x1, x2, output = self.main_net(x)
        concat_feats = [x1]

        for i in range(self.iters):
            to_concat = concat_feats + [x2]
            x_in = torch.cat(to_concat, dim=1)
            feats, x2, output = self.secondary_nets[i](x_in.detach())
            concat_feats.append(feats)

        return output
