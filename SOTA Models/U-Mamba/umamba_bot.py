"""
U-Mamba_Bot 2D – self-contained adaptation for binary segmentation.

Architecture from: U-Mamba (https://github.com/bowang-lab/U-Mamba)
  - Residual UNet encoder
  - Single Mamba layer at the bottleneck
  - UNet-style residual decoder

Only external dependency beyond PyTorch: ``mamba_ssm``
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from typing import Union, Type, List, Tuple

from mamba_ssm import Mamba


# ---------------------------------------------------------------------------
# Weight init (same as nnU-Net's InitWeights_He)
# ---------------------------------------------------------------------------
class InitWeights_He:
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


# ---------------------------------------------------------------------------
# Mamba bottleneck (fp32-safe, same as original U-Mamba)
# ---------------------------------------------------------------------------
class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)

    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.float()
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        return x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)


# ---------------------------------------------------------------------------
# Basic building blocks (re-implemented locally to avoid dynamic_network_architectures dep)
# ---------------------------------------------------------------------------
def _to_list(val):
    return list(val) if isinstance(val, (list, tuple)) else [val, val]


def _pad_for(kernel_size):
    ks = _to_list(kernel_size)
    return [k // 2 for k in ks]


class BasicResBlock(nn.Module):
    """Conv-Norm-Act × 2 with optional 1×1 shortcut."""

    def __init__(self, conv_op, input_channels, output_channels,
                 norm_op, norm_op_kwargs,
                 kernel_size=3, padding=1, stride=1,
                 use_1x1conv=False,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None):
        super().__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {"inplace": True}
        ks = _to_list(kernel_size)
        pad = _to_list(padding) if not isinstance(padding, int) else _pad_for(ks)
        st = _to_list(stride)

        self.conv1 = conv_op(input_channels, output_channels, ks, stride=st, padding=pad)
        self.norm1 = norm_op(output_channels, **norm_op_kwargs)
        self.act1 = nonlin(**nonlin_kwargs)
        self.conv2 = conv_op(output_channels, output_channels, ks, padding=pad)
        self.norm2 = norm_op(output_channels, **norm_op_kwargs)
        self.act2 = nonlin(**nonlin_kwargs)
        self.conv3 = conv_op(input_channels, output_channels, 1, stride=st) if use_1x1conv else None

    def forward(self, x):
        y = self.act1(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        if self.conv3 is not None:
            x = self.conv3(x)
        return self.act2(y + x)


class BasicBlockD(nn.Module):
    """Residual block with same in/out channels and stride=1 (replaces dynamic_network_architectures.BasicBlockD)."""

    def __init__(self, conv_op, input_channels, output_channels,
                 kernel_size=3, stride=1, conv_bias=False,
                 norm_op=None, norm_op_kwargs=None,
                 nonlin=None, nonlin_kwargs=None):
        super().__init__()
        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {"inplace": True}
        ks = _to_list(kernel_size)
        pad = _pad_for(ks)
        st = _to_list(stride)

        self.conv1 = conv_op(input_channels, output_channels, ks, stride=st, padding=pad, bias=conv_bias)
        self.norm1 = norm_op(output_channels, **norm_op_kwargs)
        self.act1 = nonlin(**nonlin_kwargs)
        self.conv2 = conv_op(output_channels, output_channels, ks, padding=pad, bias=conv_bias)
        self.norm2 = norm_op(output_channels, **norm_op_kwargs)
        self.act2 = nonlin(**nonlin_kwargs)

        if input_channels != output_channels or any(s != 1 for s in st):
            self.skip = nn.Sequential(
                conv_op(input_channels, output_channels, 1, stride=st, bias=conv_bias),
                norm_op(output_channels, **norm_op_kwargs),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act2(out + residual)


# ---------------------------------------------------------------------------
# Upsample layer (bilinear + 1×1 conv, same as original)
# ---------------------------------------------------------------------------
class UpsampleLayer(nn.Module):
    def __init__(self, conv_op, input_channels, output_channels, pool_op_kernel_size, mode="nearest"):
        super().__init__()
        self.conv = conv_op(input_channels, output_channels, kernel_size=1)
        self.pool_op_kernel_size = pool_op_kernel_size
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.pool_op_kernel_size, mode=self.mode)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------
class UNetResEncoder(nn.Module):
    def __init__(self, input_channels, n_stages, features_per_stage,
                 conv_op, kernel_sizes, strides, n_blocks_per_stage,
                 conv_bias=False, norm_op=None, norm_op_kwargs=None,
                 nonlin=None, nonlin_kwargs=None, return_skips=False):
        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [_to_list(kernel_sizes)] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [_to_list(strides)] * n_stages

        self.conv_pad_sizes = [_pad_for(k) for k in kernel_sizes]
        stem_channels = features_per_stage[0]

        self.stem = nn.Sequential(
            BasicResBlock(conv_op, input_channels, stem_channels,
                          norm_op, norm_op_kwargs,
                          kernel_size=kernel_sizes[0], padding=self.conv_pad_sizes[0],
                          stride=1, use_1x1conv=True, nonlin=nonlin, nonlin_kwargs=nonlin_kwargs),
            *[BasicBlockD(conv_op, stem_channels, stem_channels,
                          kernel_size=kernel_sizes[0], stride=1, conv_bias=conv_bias,
                          norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                          nonlin=nonlin, nonlin_kwargs=nonlin_kwargs)
              for _ in range(n_blocks_per_stage[0] - 1)]
        )

        in_ch = stem_channels
        stages = []
        for s in range(n_stages):
            stage = nn.Sequential(
                BasicResBlock(conv_op, in_ch, features_per_stage[s],
                              norm_op, norm_op_kwargs,
                              kernel_size=kernel_sizes[s], padding=self.conv_pad_sizes[s],
                              stride=strides[s], use_1x1conv=True,
                              nonlin=nonlin, nonlin_kwargs=nonlin_kwargs),
                *[BasicBlockD(conv_op, features_per_stage[s], features_per_stage[s],
                              kernel_size=kernel_sizes[s], stride=1, conv_bias=conv_bias,
                              norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                              nonlin=nonlin, nonlin_kwargs=nonlin_kwargs)
                  for _ in range(n_blocks_per_stage[s] - 1)]
            )
            stages.append(stage)
            in_ch = features_per_stage[s]

        self.stages = nn.Sequential(*stages)
        self.output_channels = features_per_stage
        self.strides = [_to_list(s) for s in strides]
        self.return_skips = return_skips
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes
        self.conv_pad_sizes_list = self.conv_pad_sizes

    def forward(self, x):
        x = self.stem(x)
        ret = []
        for s in self.stages:
            x = s(x)
            ret.append(x)
        return ret if self.return_skips else ret[-1]


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------
class UNetResDecoder(nn.Module):
    def __init__(self, encoder: UNetResEncoder, num_classes: int,
                 n_conv_per_stage, deep_supervision: bool = False):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_enc = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_enc - 1)

        stages, upsample_layers, seg_layers = [], [], []
        for s in range(1, n_stages_enc):
            feat_below = encoder.output_channels[-s]
            feat_skip = encoder.output_channels[-(s + 1)]
            stride_up = encoder.strides[-s]
            upsample_layers.append(UpsampleLayer(encoder.conv_op, feat_below, feat_skip, stride_up))

            in_ch = 2 * feat_skip if s < n_stages_enc - 1 else feat_skip
            stages.append(nn.Sequential(
                BasicResBlock(encoder.conv_op, in_ch, feat_skip,
                              encoder.norm_op, encoder.norm_op_kwargs,
                              kernel_size=encoder.kernel_sizes[-(s + 1)],
                              padding=encoder.conv_pad_sizes_list[-(s + 1)],
                              stride=1, use_1x1conv=True,
                              nonlin=encoder.nonlin, nonlin_kwargs=encoder.nonlin_kwargs),
                *[BasicBlockD(encoder.conv_op, feat_skip, feat_skip,
                              kernel_size=encoder.kernel_sizes[-(s + 1)], stride=1,
                              conv_bias=encoder.conv_bias,
                              norm_op=encoder.norm_op, norm_op_kwargs=encoder.norm_op_kwargs,
                              nonlin=encoder.nonlin, nonlin_kwargs=encoder.nonlin_kwargs)
                  for _ in range(n_conv_per_stage[s - 1] - 1)]
            ))
            seg_layers.append(encoder.conv_op(feat_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.upsample_layers = nn.ModuleList(upsample_layers)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips):
        lres = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.upsample_layers[s](lres)
            if s < len(self.stages) - 1:
                x = torch.cat((x, skips[-(s + 2)]), 1)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == len(self.stages) - 1:
                seg_outputs.append(self.seg_layers[-1](x))
            lres = x
        seg_outputs = seg_outputs[::-1]
        return seg_outputs if self.deep_supervision else seg_outputs[0]


# ---------------------------------------------------------------------------
# UMambaBot (core model)
# ---------------------------------------------------------------------------
class UMambaBot(nn.Module):
    def __init__(self, input_channels, n_stages, features_per_stage,
                 conv_op, kernel_sizes, strides,
                 n_conv_per_stage, num_classes, n_conv_per_stage_decoder,
                 conv_bias=False, norm_op=None, norm_op_kwargs=None,
                 dropout_op=None, dropout_op_kwargs=None,
                 nonlin=None, nonlin_kwargs=None,
                 deep_supervision=False, stem_channels=None):
        super().__init__()
        n_blocks = list(n_conv_per_stage) if not isinstance(n_conv_per_stage, list) else n_conv_per_stage
        n_dec = list(n_conv_per_stage_decoder) if not isinstance(n_conv_per_stage_decoder, list) else n_conv_per_stage_decoder
        for s in range(math.ceil(n_stages / 2), n_stages):
            n_blocks[s] = 1
        for s in range(math.ceil((n_stages - 1) / 2 + 0.5), n_stages - 1):
            n_dec[s] = 1

        self.encoder = UNetResEncoder(
            input_channels, n_stages, features_per_stage, conv_op,
            kernel_sizes, strides, n_blocks, conv_bias,
            norm_op, norm_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
        )
        self.mamba_layer = MambaLayer(dim=features_per_stage[-1])
        self.decoder = UNetResDecoder(self.encoder, num_classes, n_dec, deep_supervision)

    def forward(self, x):
        skips = self.encoder(x)
        skips[-1] = self.mamba_layer(skips[-1])
        return self.decoder(skips)


# ---------------------------------------------------------------------------
# Wrapper for our adapted pipeline (sigmoid output, simple constructor)
# ---------------------------------------------------------------------------
class UMambaBotAdapted(nn.Module):
    """
    Thin wrapper that builds UMambaBot with sensible 2D defaults for
    224×224 binary segmentation (matching RootBin data pipeline).
    """

    def __init__(self, num_classes=1, input_channels=3):
        super().__init__()
        n_stages = 6
        features = [32, 64, 128, 256, 320, 320]
        kernel_sizes = [[3, 3]] * n_stages
        strides = [[1, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]]
        n_conv_enc = [2, 2, 2, 2, 2, 2]
        n_conv_dec = [2, 2, 2, 2, 2]

        self.num_classes = num_classes
        self.model = UMambaBot(
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features,
            conv_op=nn.Conv2d,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_conv_per_stage=n_conv_enc,
            num_classes=num_classes,
            n_conv_per_stage_decoder=n_conv_dec,
            conv_bias=True,
            norm_op=nn.InstanceNorm2d,
            norm_op_kwargs={"eps": 1e-5, "affine": True},
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
            deep_supervision=False,
        )
        self.model.apply(InitWeights_He(1e-2))
        self._init_seg_heads()

    def _init_seg_heads(self):
        """Zero-init seg layer weights so initial logits ≈ 0 → sigmoid ≈ 0.5,
        preventing gradient explosion on the first iteration."""
        for seg in self.model.decoder.seg_layers:
            nn.init.zeros_(seg.weight)
            if seg.bias is not None:
                nn.init.zeros_(seg.bias)

    def forward(self, x):
        logits = self.model(x)
        if self.num_classes == 1:
            return torch.sigmoid(logits)
        return logits
