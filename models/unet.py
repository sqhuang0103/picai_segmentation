import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, channels, strides):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.blocks.append(ConvBlock(channels[i], channels[i + 1]))
            self.pools.append(nn.MaxPool3d(kernel_size=strides[i], stride=strides[i]))

    def forward(self, x):  # (B, in_ch, D, H, W)
        skips = []
        for block, pool in zip(self.blocks, self.pools):
            x = block(x)  # (B, out_ch, D, H, W)
            skips.append(x)
            x = pool(x)  # (B, out_ch, D//s, H//s, W//s)
        return x, skips


class Decoder(nn.Module):
    def __init__(self, channels, strides):
        super().__init__()
        self.ups = nn.ModuleList()
        self.blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.ups.append(
                nn.ConvTranspose3d(channels[i], channels[i + 1],
                                   kernel_size=strides[i], stride=strides[i])
            )
            self.blocks.append(ConvBlock(channels[i], channels[i + 1]))

    def forward(self, x, skips):  # x: (B, ch_in, D, H, W)
        for up, block, skip in zip(self.ups, self.blocks, skips):
            x = up(x)  # (B, ch_out, D*s, H*s, W*s)
            if x.shape != skip.shape:
                diff = [s - x_ for s, x_ in zip(skip.shape[2:], x.shape[2:])]
                x = nn.functional.pad(x, [0, diff[2], 0, diff[1], 0, diff[0]])
            x = torch.cat([skip, x], dim=1)  # (B, ch_out*2, D, H, W)
            x = block(x)  # (B, ch_out, D, H, W)
        return x


class UNet(nn.Module):
    """
    3D UNet following PI-CAI baseline architecture:
    - InstanceNorm + LeakyReLU
    - Anisotropic pooling strides: (2,2,2)→(1,2,2)→(1,2,2)→(1,2,2)→(2,2,2)
    - Feature channels: 32→64→128→256→512→1024
    """
    def __init__(self, in_channels=3, num_classes=2,
                 features=(32, 64, 128, 256, 512, 1024),
                 strides=((2, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2), (2, 2, 2))):
        super().__init__()
        enc_channels = [in_channels] + list(features[:-1])
        self.encoder = Encoder(enc_channels, strides)
        self.bottleneck = ConvBlock(features[-2], features[-1])

        dec_channels = list(reversed(features))
        dec_strides = list(reversed(strides))
        self.decoder = Decoder(dec_channels, dec_strides)
        self.head = nn.Conv3d(features[0], num_classes, 1)

    def forward(self, x):  # (B, C, D, H, W)
        x, skips = self.encoder(x)  # x: (B, 512, D', H', W'), skips: list of encoder outputs
        x = self.bottleneck(x)  # (B, 1024, D', H', W')
        x = self.decoder(x, skips[::-1])  # (B, 32, D, H, W)
        return self.head(x)  # (B, num_classes, D, H, W)
