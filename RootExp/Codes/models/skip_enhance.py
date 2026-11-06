"""
Skip path enhancement aligned with U-Mamba_SS2D_Conv_ASPP_EAM_SF_Bridge_adapted:
  SkipConvRefine (per-level residual conv) + SkipFusion (top-down cascade + gates).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipConvRefine(nn.Module):
    """3x3 Conv refinement on encoder skips (residual), before bridge / fusion."""

    def __init__(self, channels):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels, eps=1e-5, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels, eps=1e-5, affine=True),
        )
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        return self.act(self.refine(x) + x)


class SkipFusion(nn.Module):
    """
    Top-down cascade fusion of encoder skips (shallow -> deep order in `skip_channels`).
    Deepest tensor in the list is unchanged; shallower levels get enriched + gated residual.
    """

    def __init__(self, skip_channels, fusion_dim=64):
        super().__init__()
        n = len(skip_channels)
        self.proj_in = nn.ModuleList([nn.Conv2d(ch, fusion_dim, 1) for ch in skip_channels])
        self.fuse = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(fusion_dim * 2, fusion_dim, 1),
                    nn.InstanceNorm2d(fusion_dim, eps=1e-5, affine=True),
                    nn.LeakyReLU(inplace=True),
                )
                for _ in range(n - 1)
            ]
        )
        self.proj_out = nn.ModuleList([nn.Conv2d(fusion_dim, ch, 1) for ch in skip_channels[:-1]])
        self.gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(ch, ch, 1, bias=True),
                    nn.Sigmoid(),
                )
                for ch in skip_channels[:-1]
            ]
        )

    def forward(self, skips):
        proj = [p(s) for p, s in zip(self.proj_in, skips)]
        cascaded = [proj[-1]]
        for i in range(len(proj) - 2, -1, -1):
            deep_up = F.interpolate(cascaded[-1], size=proj[i].shape[2:], mode="bilinear", align_corners=False)
            fused = self.fuse[len(proj) - 2 - i](torch.cat([proj[i], deep_up], dim=1))
            cascaded.append(fused)
        cascaded = cascaded[::-1]

        out = []
        for i in range(len(skips) - 1):
            enriched = self.proj_out[i](cascaded[i])
            gate = self.gates[i](enriched)
            out.append(skips[i] + gate * enriched)
        out.append(skips[-1])
        return out
