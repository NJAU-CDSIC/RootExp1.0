"""
ESegFormer - Enhanced SegFormer with Edge Attention Module (EAM)
Converted from PaddlePaddle to PyTorch.
Original: SegFormer + EAM (Conv+BN+ReLU+Dilated Conv+CBAM) branch
B0 variant: embed_dims=[32, 64, 160, 256], depths=[2, 2, 2, 2]
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


# ======================== Attention Modules ========================

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
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
        attn = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return attn


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


# ======================== Transformer Components ========================

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        output = x / keep_prob * torch.floor(random_tensor + keep_prob)
        return output


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=True)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0.,
                 attn_drop=0., drop_path=0., sr_ratio=1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, attn_drop, drop, sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_ch=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=patch_size // 2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


# ======================== EAM + Backbone ========================

class EMixVisionTransformer(nn.Module):
    """
    Enhanced Mix Vision Transformer backbone.
    Standard MiT + EAM (Edge Attention Module with CBAM).
    EAM produces a 5th feature map from the input via dilated convolutions.
    """
    def __init__(self, in_ch=3, embed_dims=(32, 64, 160, 256),
                 num_heads=(1, 2, 5, 8), mlp_ratios=(4, 4, 4, 4),
                 qkv_bias=True, depths=(2, 2, 2, 2), sr_ratios=(8, 4, 2, 1),
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1):
        super().__init__()
        self.depths = depths
        self.feat_channels = list(embed_dims)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Patch embeddings
        self.patch_embed1 = OverlapPatchEmbed(7, 4, in_ch, embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(3, 2, embed_dims[0], embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(3, 2, embed_dims[1], embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(3, 2, embed_dims[2], embed_dims[3])

        # Transformer blocks
        cur = 0
        self.block1 = nn.ModuleList([
            Block(embed_dims[0], num_heads[0], mlp_ratios[0], qkv_bias, drop_rate, attn_drop_rate, dpr[cur + i], sr_ratios[0])
            for i in range(depths[0])
        ])
        self.norm1 = nn.LayerNorm(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([
            Block(embed_dims[1], num_heads[1], mlp_ratios[1], qkv_bias, drop_rate, attn_drop_rate, dpr[cur + i], sr_ratios[1])
            for i in range(depths[1])
        ])
        self.norm2 = nn.LayerNorm(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([
            Block(embed_dims[2], num_heads[2], mlp_ratios[2], qkv_bias, drop_rate, attn_drop_rate, dpr[cur + i], sr_ratios[2])
            for i in range(depths[2])
        ])
        self.norm3 = nn.LayerNorm(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([
            Block(embed_dims[3], num_heads[3], mlp_ratios[3], qkv_bias, drop_rate, attn_drop_rate, dpr[cur + i], sr_ratios[3])
            for i in range(depths[3])
        ])
        self.norm4 = nn.LayerNorm(embed_dims[3])

        # EAM: Edge Attention Module (dilated convs + CBAM)
        self.EAM = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, dilation=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=1, padding=5, dilation=5),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            CBAM(256),
        )

    def forward(self, x):
        B = x.shape[0]
        outs = []

        # EAM branch: produces c5 at 1/4 resolution
        outs.append(self.EAM(x))

        # Stage 1: 1/4
        x_t, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x_t = blk(x_t, H, W)
        x_t = self.norm1(x_t)
        x_t = x_t.reshape(B, H, W, self.feat_channels[0]).permute(0, 3, 1, 2)
        outs.append(x_t)

        # Stage 2: 1/8
        x_t, H, W = self.patch_embed2(x_t)
        for blk in self.block2:
            x_t = blk(x_t, H, W)
        x_t = self.norm2(x_t)
        x_t = x_t.reshape(B, H, W, self.feat_channels[1]).permute(0, 3, 1, 2)
        outs.append(x_t)

        # Stage 3: 1/16
        x_t, H, W = self.patch_embed3(x_t)
        for blk in self.block3:
            x_t = blk(x_t, H, W)
        x_t = self.norm3(x_t)
        x_t = x_t.reshape(B, H, W, self.feat_channels[2]).permute(0, 3, 1, 2)
        outs.append(x_t)

        # Stage 4: 1/32
        x_t, H, W = self.patch_embed4(x_t)
        for blk in self.block4:
            x_t = blk(x_t, H, W)
        x_t = self.norm4(x_t)
        x_t = x_t.reshape(B, H, W, self.feat_channels[3]).permute(0, 3, 1, 2)
        outs.append(x_t)

        return outs  # [c5_eam, c1, c2, c3, c4]


# ======================== Decoder Head ========================

class MLP_Proj(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class ESegFormer(nn.Module):
    """
    ESegFormer: Enhanced SegFormer with EAM branch.
    Backbone outputs 5 feature maps: c5 (EAM), c1, c2, c3, c4.
    Decoder fuses all 5 via MLP projections.
    """
    def __init__(self, num_classes=1, input_channels=3, embedding_dim=256,
                 embed_dims=(32, 64, 160, 256), num_heads=(1, 2, 5, 8),
                 mlp_ratios=(4, 4, 4, 4), depths=(2, 2, 2, 2),
                 sr_ratios=(8, 4, 2, 1), drop_path_rate=0.1):
        super().__init__()

        self.backbone = EMixVisionTransformer(
            in_ch=input_channels,
            embed_dims=embed_dims,
            num_heads=num_heads,
            mlp_ratios=mlp_ratios,
            depths=depths,
            sr_ratios=sr_ratios,
            drop_path_rate=drop_path_rate,
        )

        # 5 MLP projections: c5(EAM=256), c1, c2, c3, c4
        self.linear_c5 = MLP_Proj(256, embedding_dim)
        self.linear_c1 = MLP_Proj(embed_dims[0], embedding_dim)
        self.linear_c2 = MLP_Proj(embed_dims[1], embedding_dim)
        self.linear_c3 = MLP_Proj(embed_dims[2], embedding_dim)
        self.linear_c4 = MLP_Proj(embed_dims[3], embedding_dim)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embedding_dim * 5, embedding_dim, 1, bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(0.1)
        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h, w = x.shape[2:]
        feats = self.backbone(x)
        c5, c1, c2, c3, c4 = feats

        _, _, H1, W1 = c1.shape

        _c5 = self.linear_c5(c5).transpose(1, 2).reshape(-1, self.linear_c5.proj.out_features, c5.shape[2], c5.shape[3])
        _c5 = F.interpolate(_c5, size=(H1, W1), mode='bilinear', align_corners=False)

        _c4 = self.linear_c4(c4).transpose(1, 2).reshape(-1, self.linear_c4.proj.out_features, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=(H1, W1), mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).transpose(1, 2).reshape(-1, self.linear_c3.proj.out_features, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=(H1, W1), mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).transpose(1, 2).reshape(-1, self.linear_c2.proj.out_features, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=(H1, W1), mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).transpose(1, 2).reshape(-1, self.linear_c1.proj.out_features, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c5, _c4, _c3, _c2, _c1], dim=1))
        logit = self.linear_pred(self.dropout(_c))
        logit = F.interpolate(logit, size=(h, w), mode='bilinear', align_corners=False)
        return self.sigmoid(logit)
