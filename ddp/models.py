"""
U-Net architecture for MRI reconstruction.
Matches the architecture used in Notebooks 01 through 05 so existing
checkpoints (unet_4x_v2_best.pt) load directly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """U-Net for MRI reconstruction on 320x320 magnitude images.

    Dropout2d is applied after every encoder and decoder block. Inactive
    during eval() but activated for MC Dropout inference in detection.
    """

    def __init__(self, channels=(32, 64, 128, 256), dropout_p=0.05):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        self.dropout = nn.Dropout2d(p=dropout_p)

        in_ch = 1
        for ch in channels:
            self.encoders.append(ConvBlock(in_ch, ch))
            self.pools.append(nn.MaxPool2d(2))
            in_ch = ch

        self.bottleneck = ConvBlock(channels[-1], channels[-1] * 2)

        for ch in reversed(channels):
            self.upconvs.append(nn.ConvTranspose2d(ch * 2, ch, 2, stride=2))
            self.decoders.append(ConvBlock(ch * 2, ch))

        self.final = nn.Conv2d(channels[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)
            x = self.dropout(x)

        x = self.bottleneck(x)

        for upconv, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = upconv(x)
            if x.shape != skip.shape:
                x = F.pad(x, [0, skip.shape[3] - x.shape[3],
                              0, skip.shape[2] - x.shape[2]])
            x = torch.cat([x, skip], dim=1)
            x = dec(x)
            x = self.dropout(x)

        return self.final(x)
