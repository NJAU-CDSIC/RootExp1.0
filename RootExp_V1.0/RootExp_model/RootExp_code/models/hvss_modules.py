"""
H-vmunet HVSS building blocks (H_SS2D + Block + LayerNorm) for use after Conv.
SS2D implementation from local vmamba.py.

Lite defaults: smaller H_SS2D ``order`` (fewer parallel SS2D scales), lower
``d_state``, and narrower MLP — most params come from H_SS2D, not ``depth``.
"""
import os
import sys
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath

from models.vmamba import SS2D


class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, padding=1, stride=1, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            dim_in, dim_in, kernel_size=kernel_size, padding=padding,
            stride=stride, dilation=dilation, groups=dim_in,
        )
        self.norm_layer = nn.GroupNorm(4, dim_in)
        self.conv2 = nn.Conv2d(dim_in, dim_out, kernel_size=1)

    def forward(self, x):
        return self.conv2(self.norm_layer(self.conv1(x)))


if "DWCONV_IMPL" in os.environ:
    try:
        sys.path.append(os.environ["DWCONV_IMPL"])
        from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM

        def get_dwconv(dim, kernel, bias):
            return DepthWiseConv2dImplicitGEMM(dim, kernel, bias)
    except Exception:
        print(traceback.format_exc())

        def get_dwconv(dim, kernel, bias):
            return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel - 1) // 2, bias=bias, groups=dim)
else:
    def get_dwconv(dim, kernel, bias):
        return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel - 1) // 2, bias=bias, groups=dim)


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class H_SS2D(nn.Module):
    def __init__(self, dim, order=5, gflayer=None, h=14, w=8, s=1.0, d_state=16):
        super().__init__()
        self.order = order
        self.dims = [dim // 2 ** i for i in range(order)]
        self.dims.reverse()
        self.proj_in = nn.Conv2d(dim, 2 * dim, 1)

        if gflayer is None:
            self.dwconv = get_dwconv(sum(self.dims), 7, True)
        else:
            self.dwconv = gflayer(sum(self.dims), h=h, w=w)

        self.proj_out = nn.Conv2d(dim, dim, 1)
        self.pws = nn.ModuleList(
            [nn.Conv2d(self.dims[i], self.dims[i + 1], 1) for i in range(order - 1)]
        )

        num = len(self.dims)
        if num == 2:
            self.ss2d_1 = SS2D(d_model=self.dims[1], dropout=0, d_state=d_state)
        elif num == 3:
            self.ss2d_1 = SS2D(d_model=self.dims[1], dropout=0, d_state=d_state)
            self.ss2d_2 = SS2D(d_model=self.dims[2], dropout=0, d_state=d_state)
        elif num == 4:
            self.ss2d_1 = SS2D(d_model=self.dims[1], dropout=0, d_state=d_state)
            self.ss2d_2 = SS2D(d_model=self.dims[2], dropout=0, d_state=d_state)
            self.ss2d_3 = SS2D(d_model=self.dims[3], dropout=0, d_state=d_state)
        elif num == 5:
            self.ss2d_1 = SS2D(d_model=self.dims[1], dropout=0, d_state=d_state)
            self.ss2d_2 = SS2D(d_model=self.dims[2], dropout=0, d_state=d_state)
            self.ss2d_3 = SS2D(d_model=self.dims[3], dropout=0, d_state=d_state)
            self.ss2d_4 = SS2D(d_model=self.dims[4], dropout=0, d_state=d_state)

        self.ss2d_in = SS2D(d_model=self.dims[0], dropout=0, d_state=d_state)
        self.scale = s

    def forward(self, x, mask=None, dummy=False):
        fused_x = self.proj_in(x)
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)
        dw_abc = self.dwconv(abc) * self.scale
        dw_list = torch.split(dw_abc, self.dims, dim=1)
        x = pwa * dw_list[0]
        x = x.permute(0, 2, 3, 1)
        x = self.ss2d_in(x)
        x = x.permute(0, 3, 1, 2)

        for i in range(self.order - 1):
            x = self.pws[i](x) * dw_list[i + 1]
            x = x.permute(0, 2, 3, 1)
            if i == 0:
                x = self.ss2d_1(x)
            elif i == 1:
                x = self.ss2d_2(x)
            elif i == 2:
                x = self.ss2d_3(x)
            elif i == 3:
                x = self.ss2d_4(x)
            x = x.permute(0, 3, 1, 2)

        return self.proj_out(x)


class Block(nn.Module):
    """H-vmunet H_VSS block: LayerNorm → H_SS2D → residual; then MLP branch."""

    def __init__(
        self,
        dim,
        drop_path=0.0,
        layer_scale_init_value=1e-6,
        hvss_order=2,
        hvss_d_state=6,
        mlp_ratio=1.5,
    ):
        super().__init__()
        self.norm1 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.H_SS2D = H_SS2D(dim, order=hvss_order, d_state=hvss_d_state)
        hidden = max(int(dim * mlp_ratio), 32)
        self.norm2 = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden, dim)
        self.gamma1 = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.gamma2 = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        gamma1 = self.gamma1.view(C, 1, 1) if self.gamma1 is not None else 1
        x = x + self.drop_path(gamma1 * self.H_SS2D(self.norm1(x)))
        inp = x
        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma2 is not None:
            x = self.gamma2 * x
        x = x.permute(0, 3, 1, 2)
        return inp + self.drop_path(x)


def make_hvss_stack(
    dim,
    depth=1,
    drop_path_rate=0.1,
    hvss_order=2,
    hvss_d_state=6,
    mlp_ratio=1.5,
):
    if depth <= 0:
        return nn.Identity()
    dpr = [x.item() for x in torch.linspace(0, drop_path_rate, max(depth, 1))]
    layers = []
    for j in range(depth):
        layers.append(
            Block(
                dim,
                drop_path=dpr[j],
                layer_scale_init_value=1e-6,
                hvss_order=hvss_order,
                hvss_d_state=hvss_d_state,
                mlp_ratio=mlp_ratio,
            )
        )
    return nn.Sequential(*layers)
