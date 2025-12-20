#!/usr/bin/python

from torch import nn

class IdentityConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                groups=1,
            )
        self.bn1 = nn.BatchNorm2d(out_channels)

    def reset_parameters(self):
        self.bn1.reset_parameters()
        self.conv1.reset_parameters()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        return x
