#!/usr/bin/python

from torch import nn

class SqueezeExcitation(nn.Module):
   def __init__(self, in_channels, reduction_channels):
      super().__init__()
      self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduction_channels, kernel_size=1, stride=1,),
            nn.SiLU(inplace=True),
            nn.Conv2d(reduction_channels, in_channels, kernel_size=1, stride=1,),
            nn.Sigmoid()
         )

   def reset_parameters(self):
      for layer in self.block:
         layer.reset_parameters()

   def forward(self, x):
      return x*self.block(x)
