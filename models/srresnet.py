import math
import torch
import torch.nn as nn

# -----------------------------
# Residual Block
# -----------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels: int = 64):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

# -----------------------------
# Upsample Block
# -----------------------------
class UpsampleBlock(nn.Module):
    def __init__(self, channels: int = 64, scale_factor: int = 2):
        super(UpsampleBlock, self).__init__()
        self.upsample = nn.Sequential(
            nn.Conv2d(channels, channels * scale_factor ** 2, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.PReLU()
        )

    def forward(self, x):
        return self.upsample(x)

# -----------------------------
# SRResNet
# -----------------------------
class SRResNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3,
                 channels: int = 64, num_residual_blocks: int = 16,
                 upscale_factor: int = 4):
        super(SRResNet, self).__init__()

        # Initial conv layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )

        # Residual blocks
        trunk = [ResidualBlock(channels) for _ in range(num_residual_blocks)]
        self.trunk = nn.Sequential(*trunk)

        # Conv after residual blocks
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels)
        )

        # Upsampling blocks
        upsample_blocks = []
        for _ in range(int(math.log2(upscale_factor))):
            upsample_blocks.append(UpsampleBlock(channels, scale_factor=2))
        self.upsample = nn.Sequential(*upsample_blocks)

        # Final reconstruction layer
        self.conv3 = nn.Conv2d(channels, out_channels, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.trunk(x1)
        x2 = self.conv2(x2)
        x = x1 + x2
        x = self.upsample(x)
        x = self.conv3(x)
        x = torch.tanh(x)
        #x = torch.clamp(x, -1.0, 1.0)  # keep output in [0,1] for your dataset
        return x

