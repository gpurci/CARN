#!/usr/bin/python

from torch import nn

class InputStride(nn.Module):
    def __init__(self, img_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
                img_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=0,
                bias=False,
                groups=img_channels,
            )
        self.bn1 = nn.BatchNorm2d(out_channels)

    def reset_parameters(self):
        self.bn1.reset_parameters()
        self.conv1.reset_parameters()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        return x
