import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class DWConv(nn.Module):
    """Depth-wise convolution for FFN"""
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=True)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MLP_FFN(nn.Module):
    """Mix-FFN with depth-wise convolution"""
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


class EfficientSelfAttention(nn.Module):
    """Efficient Multi-Head Attention with spatial reduction"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        self.num_heads = num_heads
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


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., sr_ratio=1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(dim, num_heads, qkv_bias, attn_drop, drop, sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP_FFN(dim, int(dim * mlp_ratio), drop=drop)
        self.drop_path = nn.Identity() if drop_path <= 0. else DropPath(drop_path)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


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
        random_tensor = torch.floor(random_tensor + keep_prob)
        output = x / keep_prob * random_tensor
        return output


class OverlapPatchEmbed(nn.Module):
    """Overlapping patch embedding"""
    def __init__(self, patch_size=7, stride=4, in_ch=3, embed_dim=64):
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


class MixTransformerEncoder(nn.Module):
    """Mix Transformer Encoder (MiT) - SegFormer backbone"""
    def __init__(self, in_ch=3, embed_dims=(32, 64, 160, 256), num_heads=(1, 2, 5, 8),
                 mlp_ratios=(4, 4, 4, 4), depths=(2, 2, 2, 2), sr_ratios=(8, 4, 2, 1),
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1):
        super().__init__()
        self.depths = depths

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Stage 1: 1/4
        self.patch_embed1 = OverlapPatchEmbed(7, 4, in_ch, embed_dims[0])
        self.block1 = nn.ModuleList([
            TransformerBlock(embed_dims[0], num_heads[0], mlp_ratios[0], True, drop_rate, attn_drop_rate,
                             dpr[sum(depths[:0]) + i], sr_ratios[0])
            for i in range(depths[0])
        ])
        self.norm1 = nn.LayerNorm(embed_dims[0])

        # Stage 2: 1/8
        self.patch_embed2 = OverlapPatchEmbed(3, 2, embed_dims[0], embed_dims[1])
        self.block2 = nn.ModuleList([
            TransformerBlock(embed_dims[1], num_heads[1], mlp_ratios[1], True, drop_rate, attn_drop_rate,
                             dpr[sum(depths[:1]) + i], sr_ratios[1])
            for i in range(depths[1])
        ])
        self.norm2 = nn.LayerNorm(embed_dims[1])

        # Stage 3: 1/16
        self.patch_embed3 = OverlapPatchEmbed(3, 2, embed_dims[1], embed_dims[2])
        self.block3 = nn.ModuleList([
            TransformerBlock(embed_dims[2], num_heads[2], mlp_ratios[2], True, drop_rate, attn_drop_rate,
                             dpr[sum(depths[:2]) + i], sr_ratios[2])
            for i in range(depths[2])
        ])
        self.norm3 = nn.LayerNorm(embed_dims[2])

        # Stage 4: 1/32
        self.patch_embed4 = OverlapPatchEmbed(3, 2, embed_dims[2], embed_dims[3])
        self.block4 = nn.ModuleList([
            TransformerBlock(embed_dims[3], num_heads[3], mlp_ratios[3], True, drop_rate, attn_drop_rate,
                             dpr[sum(depths[:3]) + i], sr_ratios[3])
            for i in range(depths[3])
        ])
        self.norm4 = nn.LayerNorm(embed_dims[3])

    def forward(self, x):
        outs = []

        # Stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(-1, H, W, x.shape[-1]).permute(0, 3, 1, 2)
        outs.append(x)

        # Stage 2
        x, H, W = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(-1, H, W, x.shape[-1]).permute(0, 3, 1, 2)
        outs.append(x)

        # Stage 3
        x, H, W = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(-1, H, W, x.shape[-1]).permute(0, 3, 1, 2)
        outs.append(x)

        # Stage 4
        x, H, W = self.patch_embed4(x)
        for blk in self.block4:
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(-1, H, W, x.shape[-1]).permute(0, 3, 1, 2)
        outs.append(x)

        return outs


class SegFormerHead(nn.Module):
    """All-MLP decoder head for SegFormer"""
    def __init__(self, in_channels=(32, 64, 160, 256), embed_dim=256, num_classes=1):
        super().__init__()
        self.linear_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, embed_dim, 1),
                nn.BatchNorm2d(embed_dim),
            ) for ch in in_channels
        ])
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(0.1)
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, 1)

    def forward(self, features):
        B, _, H, W = features[0].shape
        outs = []
        for i, (feat, linear) in enumerate(zip(features, self.linear_layers)):
            x = linear(feat)
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
            outs.append(x)

        x = self.linear_fuse(torch.cat(outs, dim=1))
        x = self.dropout(x)
        x = self.linear_pred(x)
        return x


class SegFormer(nn.Module):
    """
    SegFormer - Xie et al. 2021 (NeurIPS)
    MiT-B0 variant: lightweight hierarchical transformer encoder + MLP decoder
    """
    def __init__(self, num_classes=1, input_channels=3, embed_dims=(32, 64, 160, 256),
                 num_heads=(1, 2, 5, 8), mlp_ratios=(4, 4, 4, 4), depths=(2, 2, 2, 2),
                 sr_ratios=(8, 4, 2, 1), decoder_dim=256, drop_path_rate=0.1):
        super().__init__()
        self.encoder = MixTransformerEncoder(
            in_ch=input_channels,
            embed_dims=embed_dims,
            num_heads=num_heads,
            mlp_ratios=mlp_ratios,
            depths=depths,
            sr_ratios=sr_ratios,
            drop_path_rate=drop_path_rate,
        )
        self.decoder = SegFormerHead(
            in_channels=embed_dims,
            embed_dim=decoder_dim,
            num_classes=num_classes,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h, w = x.shape[2:]
        features = self.encoder(x)
        out = self.decoder(features)
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
        return self.sigmoid(out)
