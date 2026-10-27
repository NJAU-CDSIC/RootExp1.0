import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_ch, out_ch, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv3 = nn.Conv2d(out_ch, out_ch * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNetBackbone(nn.Module):
    """ResNet-50 backbone with dilated convolutions for DeepLabV3+"""
    def __init__(self, input_channels=3, output_stride=16):
        super().__init__()
        layers = [3, 4, 6, 3]  # ResNet-50

        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise ValueError(f'Unsupported output_stride: {output_stride}')

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.layer1 = self._make_layer(Bottleneck, 64, layers[0], stride=strides[0], dilation=dilations[0])
        self.layer2 = self._make_layer(Bottleneck, 128, layers[1], stride=strides[1], dilation=dilations[1])
        self.layer3 = self._make_layer(Bottleneck, 256, layers[2], stride=strides[2], dilation=dilations[2])
        self.layer4 = self._make_layer(Bottleneck, 512, layers[3], stride=strides[3], dilation=dilations[3])

    def _make_layer(self, block, channels, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, channels * block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(channels * block.expansion),
            )

        layer_list = [block(self.in_channels, channels, stride, dilation, downsample)]
        self.in_channels = channels * block.expansion
        for _ in range(1, blocks):
            layer_list.append(block(self.in_channels, channels, dilation=dilation))

        return nn.Sequential(*layer_list)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        low_level = self.layer1(x)  # 1/4
        x = self.layer2(low_level)  # 1/8
        x = self.layer3(x)          # 1/16
        x = self.layer4(x)          # 1/16 (dilated)
        return x, low_level


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
    def __init__(self, in_ch, out_ch=256, rates=(6, 12, 18)):
        super().__init__()
        modules = [
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        ]
        for rate in rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ))
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * (len(rates) + 2), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        gp = self.global_pool(x)
        gp = F.interpolate(gp, size=x.shape[2:], mode='bilinear', align_corners=True)
        res.append(gp)
        x = torch.cat(res, dim=1)
        return self.project(x)


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ - Chen et al. 2018
    ResNet-50 backbone + ASPP + Decoder
    """
    def __init__(self, num_classes=1, input_channels=3, output_stride=16):
        super().__init__()
        self.backbone = ResNetBackbone(input_channels, output_stride)
        self.aspp = ASPP(2048, 256)

        self.low_level_conv = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h, w = x.shape[2:]
        encoder_out, low_level = self.backbone(x)
        aspp_out = self.aspp(encoder_out)

        aspp_out = F.interpolate(aspp_out, size=low_level.shape[2:], mode='bilinear', align_corners=True)
        low_level = self.low_level_conv(low_level)

        x = torch.cat([aspp_out, low_level], dim=1)
        x = self.decoder(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        return self.sigmoid(x)
