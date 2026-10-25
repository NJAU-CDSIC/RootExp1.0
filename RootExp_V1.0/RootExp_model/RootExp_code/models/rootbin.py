import torch
import torch.nn as nn
import torch.nn.functional as F

from models.eam_module import EAM
from models.sc_att_bridge import SC_Att_Bridge
from models.hvss_modules import make_hvss_stack
from models.skip_enhance import SkipConvRefine, SkipFusion

ENCODER_CHANNELS_WIDE = (64, 128, 256, 512, 1024)
ENCODER_CHANNELS_ULTRA_LITE = (32, 64, 128, 256, 384)
ENCODER_CHANNELS_LITE_DEFAULT = (40, 80, 160, 320, 640)


def _parse_encoder_channels(encoder_channels):
    if encoder_channels is None:
        return ENCODER_CHANNELS_LITE_DEFAULT
    ch = tuple(int(x) for x in encoder_channels)
    if len(ch) != 5:
        raise ValueError("encoder_channels must be a sequence of 5 ints [c0..c4].")
    if any(x <= 0 for x in ch):
        raise ValueError("encoder_channels must be positive.")
    return ch


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.pool_conv(x)


class DownHvss(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        hvss_depth=1,
        drop_path_rate=0.1,
        hvss_order=2,
        hvss_d_state=6,
        mlp_ratio=1.5,
    ):
        super().__init__()
        self.pool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))
        self.hvss = make_hvss_stack(
            out_ch,
            depth=hvss_depth,
            drop_path_rate=drop_path_rate,
            hvss_order=hvss_order,
            hvss_d_state=hvss_d_state,
            mlp_ratio=mlp_ratio,
        )

    def forward(self, x):
        x = self.pool_conv(x)
        return self.hvss(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class UpWithHvss(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        use_hvss=False,
        hvss_depth=1,
        drop_path_rate=0.1,
        hvss_order=2,
        hvss_d_state=6,
        mlp_ratio=1.5,
    ):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)
        self.hvss = (
            make_hvss_stack(
                out_ch,
                depth=hvss_depth,
                drop_path_rate=drop_path_rate,
                hvss_order=hvss_order,
                hvss_d_state=hvss_d_state,
                mlp_ratio=mlp_ratio,
            )
            if use_hvss
            else nn.Identity()
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return self.hvss(x)


class ASPP(nn.Module):
    """Light ASPP at bottleneck. Requires out_ch % 4 == 0."""

    def __init__(self, in_ch, out_ch, dilations=(1, 6, 12, 18)):
        super().__init__()
        inter = out_ch // 4
        branches = []
        for d in dilations:
            if d == 1:
                branches.append(
                    nn.Sequential(
                        nn.Conv2d(in_ch, inter, 1, bias=False),
                        nn.BatchNorm2d(inter),
                        nn.ReLU(inplace=True),
                    )
                )
            else:
                branches.append(
                    nn.Sequential(
                        nn.Conv2d(in_ch, inter, 3, padding=d, dilation=d, bias=False),
                        nn.BatchNorm2d(inter),
                        nn.ReLU(inplace=True),
                    )
                )
        self.branches = nn.ModuleList(branches)
        self.project = nn.Sequential(
            nn.Conv2d(inter * len(dilations), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        feats = [b(x) for b in self.branches]
        return self.project(torch.cat(feats, dim=1))


class RootBinBlockMambaEAMASPPNet(nn.Module):
    """
    RootBin_Block_Mamba + ASPP then EAM fusion at bottleneck before SC_Att_Bridge.
    Order: x5 -> ASPP (multi-scale context) -> concat EAM(image) -> 1x1 fuse -> bridge.
    """

    def __init__(
        self,
        num_classes=1,
        input_channels=3,
        stage_name="stage0_unet_scblock_hvss_eam_aspp",
        split_att="fc",
        encoder_channels=None,
        use_eam=True,
        use_aspp=True,
        use_hvss=True,
        use_bridge=True,
        hvss_depth=1,
        drop_path_rate=0.1,
        hvss_order=2,
        hvss_d_state=6,
        mlp_ratio=1.5,
        use_skip_enhance=True,
        skip_fusion_dim=64,
        aspp_dilations=(1, 6, 12, 18),
    ):
        super().__init__()
        self.stage_name = stage_name
        self.use_eam = use_eam
        self.use_aspp = use_aspp
        self.use_hvss = use_hvss
        self.use_bridge = use_bridge
        self.use_skip_enhance = use_skip_enhance
        c0, c1, c2, c3, c4 = _parse_encoder_channels(encoder_channels)
        if use_aspp and c4 % 4 != 0:
            raise ValueError(f"bottleneck c4={c4} must be divisible by 4 for ASPP.")
        self.encoder_channels = (c0, c1, c2, c3, c4)

        self.inc = DoubleConv(input_channels, c0)
        self.down1 = Down(c0, c1)
        self.down2 = Down(c1, c2)
        if self.use_hvss:
            self.down3 = DownHvss(
                c2, c3,
                hvss_depth=hvss_depth,
                drop_path_rate=drop_path_rate,
                hvss_order=hvss_order,
                hvss_d_state=hvss_d_state,
                mlp_ratio=mlp_ratio,
            )
        else:
            self.down3 = Down(c2, c3)
        self.down4 = Down(c3, c4)
        self.aspp = ASPP(c4, c4, dilations=aspp_dilations) if use_aspp else None

        if self.use_eam:
            self.eam = EAM(in_channels=input_channels, out_channels=c4)
            self.eam_fusion = nn.Sequential(
                nn.Conv2d(c4 * 2, c4, 1, bias=False),
                nn.InstanceNorm2d(c4, eps=1e-5, affine=True),
                nn.LeakyReLU(inplace=True),
            )
        else:
            self.eam = None
            self.eam_fusion = None

        if self.use_bridge and self.use_skip_enhance:
            self.skip_refines = nn.ModuleList(
                [SkipConvRefine(c) for c in (c0, c1, c2, c3)]
            )
            self.bridge_mid = SC_Att_Bridge([c1, c2, c3], split_att=split_att)
            self.skip_fusion = SkipFusion([c1, c2, c3], fusion_dim=skip_fusion_dim)
            self.bridge = None
        elif self.use_bridge:
            self.skip_refines = None
            self.bridge_mid = None
            self.skip_fusion = None
            self.bridge = SC_Att_Bridge(list(self.encoder_channels), split_att=split_att)
        else:
            self.skip_refines = None
            self.bridge_mid = None
            self.skip_fusion = None
            self.bridge = None
        self.up1 = UpWithHvss(
            c4 + c3, c3, use_hvss=self.use_hvss,
            hvss_depth=hvss_depth,
            drop_path_rate=drop_path_rate,
            hvss_order=hvss_order,
            hvss_d_state=hvss_d_state,
            mlp_ratio=mlp_ratio,
        )
        self.up2 = Up(c3 + c2, c2)
        self.up3 = Up(c2 + c1, c1)
        self.up4 = Up(c1 + c0, c0)
        self.outc = nn.Conv2d(c0, num_classes, 1)

    def forward(self, x):
        x_in = x
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        if self.use_aspp and self.aspp is not None:
            x5 = self.aspp(x5)

        if self.use_eam and self.eam is not None:
            eam_feat = self.eam(x_in)
            if eam_feat.shape[2:] != x5.shape[2:]:
                eam_feat = F.interpolate(
                    eam_feat, size=x5.shape[2:], mode="bilinear", align_corners=False
                )
            x5 = self.eam_fusion(torch.cat([x5, eam_feat], dim=1))

        if self.use_bridge and self.use_skip_enhance:
            x1 = self.skip_refines[0](x1)
            x2 = self.skip_refines[1](x2)
            x3 = self.skip_refines[2](x3)
            x4 = self.skip_refines[3](x4)
            x2, x3, x4 = self.bridge_mid([x2, x3, x4])
            x2, x3, x4 = self.skip_fusion([x2, x3, x4])
        elif self.use_bridge:
            x1, x2, x3, x4, x5 = self.bridge([x1, x2, x3, x4, x5])
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return torch.sigmoid(x)
