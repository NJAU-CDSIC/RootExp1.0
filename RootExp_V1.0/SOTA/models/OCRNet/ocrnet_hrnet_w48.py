"""OCRNet (Yuan et al., ECCV 2020): HRNetV2-W48 + OCR head.

Matches official HRNet-OCR assembly (mid_channels=512, key_channels=256).
Published size: ~70.3M params (Cityscapes / HRNet-OCR README).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

BN_MOMENTUM = 0.1


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels, num_channels, multi_scale_output=True):
        super().__init__()
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output
        self.num_inchannels = num_inchannels
        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1, stride=stride, bias=False,
                ),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion, momentum=BN_MOMENTUM),
            )
        layers = [block(self.num_inchannels[branch_index], num_channels[branch_index], stride, downsample)]
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for _ in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index]))
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        return nn.ModuleList([self._make_one_branch(i, block, num_blocks, num_channels) for i in range(num_branches)])

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None
        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j], num_inchannels[i], 1, 1, 0, bias=False),
                        nn.BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM),
                    ))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j], num_inchannels[i], 3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM),
                            ))
                        else:
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j], num_inchannels[j], 3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_inchannels[j], momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=True),
                            ))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=x[i].shape[2:],
                        mode='bilinear', align_corners=True,
                    )
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


class HighResolutionNet(nn.Module):
    """HRNetV2-W48 backbone (segmentation fuse → 720-d)."""

    def __init__(self, in_chans: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, 64, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, 3, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 64, 4)
        block = BasicBlock
        num_channels = [c * block.expansion for c in [48, 96]]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre = self._make_stage(1, 2, [4, 4], [48, 96], num_channels)
        num_channels = [c * block.expansion for c in [48, 96, 192]]
        self.transition2 = self._make_transition_layer(pre, num_channels)
        self.stage3, pre = self._make_stage(4, 3, [4, 4, 4], [48, 96, 192], num_channels)
        num_channels = [c * block.expansion for c in [48, 96, 192, 384]]
        self.transition3 = self._make_transition_layer(pre, num_channels)
        self.stage4, pre = self._make_stage(3, 4, [4, 4, 4, 4], [48, 96, 192, 384], num_channels, True)
        self.last_inp_channels = int(sum(pre))

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )
        layers = [block(inplanes, planes, stride, downsample)]
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def _make_transition_layer(self, num_channels_pre, num_channels_cur):
        num_branches_cur = len(num_channels_cur)
        num_branches_pre = len(num_channels_pre)
        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur[i] != num_channels_pre[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre[i], num_channels_cur[i], 3, 1, 1, bias=False),
                        nn.BatchNorm2d(num_channels_cur[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True),
                    ))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre[-1]
                    outchannels = num_channels_cur[i] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True),
                    ))
                transition_layers.append(nn.Sequential(*conv3x3s))
        return nn.ModuleList(transition_layers)

    def _make_stage(self, num_modules, num_branches, num_blocks, num_channels, num_inchannels, multi_scale_output=True):
        modules = []
        for i in range(num_modules):
            reset = multi_scale_output or (i != num_modules - 1)
            modules.append(HighResolutionModule(
                num_branches, BasicBlock, num_blocks, num_inchannels, num_channels, reset))
            num_inchannels = modules[-1].get_num_inchannels()
        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.layer1(x)
        x_list = []
        for i in range(2):
            x_list.append(self.transition1[i](x) if self.transition1[i] is not None else x)
        y_list = self.stage2(x_list)
        x_list = []
        for i in range(3):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[i] if i < len(y_list) else y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        x_list = []
        for i in range(4):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[i] if i < len(y_list) else y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)
        h, w = y_list[0].shape[2:]
        feats = [y_list[0]] + [
            F.interpolate(y_list[i], size=(h, w), mode='bilinear', align_corners=True) for i in range(1, 4)
        ]
        return torch.cat(feats, 1)


class SpatialGather(nn.Module):
    def __init__(self, scale=1):
        super().__init__()
        self.scale = scale

    def forward(self, feats, probs):
        b, c, _, _ = probs.shape
        probs = F.softmax(self.scale * probs.view(b, c, -1), dim=2)
        feats = feats.view(b, feats.size(1), -1).permute(0, 2, 1)
        return torch.matmul(probs, feats).permute(0, 2, 1).unsqueeze(3)


class ObjectAttentionBlock(nn.Module):
    def __init__(self, in_channels, key_channels):
        super().__init__()
        self.key_channels = key_channels
        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels, key_channels, 1, bias=False), nn.BatchNorm2d(key_channels), nn.ReLU(inplace=True),
            nn.Conv2d(key_channels, key_channels, 1, bias=False), nn.BatchNorm2d(key_channels), nn.ReLU(inplace=True),
        )
        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels, key_channels, 1, bias=False), nn.BatchNorm2d(key_channels), nn.ReLU(inplace=True),
            nn.Conv2d(key_channels, key_channels, 1, bias=False), nn.BatchNorm2d(key_channels), nn.ReLU(inplace=True),
        )
        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels, key_channels, 1, bias=False), nn.BatchNorm2d(key_channels), nn.ReLU(inplace=True),
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(key_channels, in_channels, 1, bias=False), nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x, proxy):
        b, _, h, w = x.shape
        query = self.f_pixel(x).view(b, self.key_channels, -1).permute(0, 2, 1)
        key = self.f_object(proxy).view(b, self.key_channels, -1)
        value = self.f_down(proxy).view(b, self.key_channels, -1).permute(0, 2, 1)
        sim = F.softmax((self.key_channels ** -0.5) * torch.matmul(query, key), dim=-1)
        context = torch.matmul(sim, value).permute(0, 2, 1).contiguous().view(b, self.key_channels, h, w)
        return self.f_up(context)


class SpatialOCR(nn.Module):
    def __init__(self, in_channels, key_channels, out_channels, dropout=0.05):
        super().__init__()
        self.object_context = ObjectAttentionBlock(in_channels, key_channels)
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(2 * in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

    def forward(self, feats, proxy):
        return self.conv_bn_dropout(torch.cat([self.object_context(feats, proxy), feats], 1))


class OCRNet(nn.Module):
    """HRNetV2-W48 + OCR (paper / official HRNet-OCR defaults)."""

    def __init__(self, num_classes: int = 2, in_chans: int = 3, mid: int = 512, key: int = 256):
        super().__init__()
        self.backbone = HighResolutionNet(in_chans=in_chans)
        in_ch = self.backbone.last_inp_channels
        self.conv3x3_ocr = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, 1, 1, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
        )
        self.ocr_gather = SpatialGather(scale=1)
        self.ocr_distri = SpatialOCR(mid, key, mid, dropout=0.05)
        self.cls_head = nn.Conv2d(mid, num_classes, 1)
        self.aux_head = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1, bias=False),
            nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, num_classes, 1),
        )

    def forward(self, x):
        feats = self.backbone(x)
        out_aux = self.aux_head(feats)
        feats_ocr = self.conv3x3_ocr(feats)
        feats_ocr = self.ocr_distri(feats_ocr, self.ocr_gather(feats_ocr, out_aux))
        out = self.cls_head(feats_ocr)
        return F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=True)


def build_ocrnet(num_classes: int = 2, in_chans: int = 3) -> OCRNet:
    return OCRNet(num_classes=num_classes, in_chans=in_chans)
