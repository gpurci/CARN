#!/usr/bin/python

from torch import nn

class ResNetInput(nn.Module):
   def __init__(self, img_channels, out_channels):
      super().__init__()
      self.conv1 = nn.Conv2d(
            img_channels,
            out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
            groups=1,
         )
      self.bn1      = nn.BatchNorm2d(out_channels)
      self.activ_fn = nn.SiLU(inplace=True)
      self.maxpool  = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

   def reset_parameters(self):
      self.bn1.reset_parameters()
      self.conv1.reset_parameters()

   def forward(self, x):
      x = self.conv1(x)
      x = self.bn1(x)
      x = self.activ_fn(x)
      x = self.maxpool(x)
      return x
