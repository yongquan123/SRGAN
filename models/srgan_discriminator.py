import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super(Discriminator, self).__init__()

        def conv_block(in_ch, out_ch, stride):
            layers = [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            conv_block(base_channels, base_channels, stride=2),
            conv_block(base_channels, base_channels*2, stride=1),
            conv_block(base_channels*2, base_channels*2, stride=2),
            conv_block(base_channels*2, base_channels*4, stride=1),
            conv_block(base_channels*4, base_channels*4, stride=2),
            conv_block(base_channels*4, base_channels*8, stride=1),
            conv_block(base_channels*8, base_channels*8, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels*8, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x)
        return x
