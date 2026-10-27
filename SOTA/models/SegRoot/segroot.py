"""
SegNet - Classic VGG16-based encoder-decoder with indices-guided upsampling
Badrinarayanan et al. 2017 - "SegNet: A Deep Convolutional Encoder-Decoder Architecture"
Parameter count ~29.6M matching the paper reference.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class SegNet(nn.Module):
    """
    Classic SegNet with VGG16-style encoder and symmetric decoder.
    Encoder: 5 stages (2,2,3,3,3 conv layers) matching VGG16
    Decoder: symmetric 5 stages with max-unpool
    """
    def __init__(self, num_classes=1, input_channels=3):
        super().__init__()

        # Encoder (VGG16 style)
        self.enc1 = nn.Sequential(ConvBNReLU(input_channels, 64), ConvBNReLU(64, 64))
        self.enc2 = nn.Sequential(ConvBNReLU(64, 128), ConvBNReLU(128, 128))
        self.enc3 = nn.Sequential(ConvBNReLU(128, 256), ConvBNReLU(256, 256), ConvBNReLU(256, 256))
        self.enc4 = nn.Sequential(ConvBNReLU(256, 512), ConvBNReLU(512, 512), ConvBNReLU(512, 512))
        self.enc5 = nn.Sequential(ConvBNReLU(512, 512), ConvBNReLU(512, 512), ConvBNReLU(512, 512))

        # Decoder (symmetric)
        self.dec5 = nn.Sequential(ConvBNReLU(512, 512), ConvBNReLU(512, 512), ConvBNReLU(512, 512))
        self.dec4 = nn.Sequential(ConvBNReLU(512, 512), ConvBNReLU(512, 512), ConvBNReLU(512, 256))
        self.dec3 = nn.Sequential(ConvBNReLU(256, 256), ConvBNReLU(256, 256), ConvBNReLU(256, 128))
        self.dec2 = nn.Sequential(ConvBNReLU(128, 128), ConvBNReLU(128, 64))
        self.dec1 = nn.Sequential(ConvBNReLU(64, 64), nn.Conv2d(64, num_classes, 1))

        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encode
        x = self.enc1(x)
        x, idx1 = self.pool(x)
        x = self.enc2(x)
        x, idx2 = self.pool(x)
        x = self.enc3(x)
        x, idx3 = self.pool(x)
        x = self.enc4(x)
        x, idx4 = self.pool(x)
        x = self.enc5(x)
        x, idx5 = self.pool(x)

        # Decode
        x = self.unpool(x, idx5)
        x = self.dec5(x)
        x = self.unpool(x, idx4)
        x = self.dec4(x)
        x = self.unpool(x, idx3)
        x = self.dec3(x)
        x = self.unpool(x, idx2)
        x = self.dec2(x)
        x = self.unpool(x, idx1)
        x = self.dec1(x)

        return self.sigmoid(x)
