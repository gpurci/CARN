#!/usr/bin/python

from torch import nn

from sys_function import * # este in root
sys_remove_modules("layers.dropout.stochastic_depth")

from layers.excitation.squeeze_excitation_2d import *
from layers.dropout.stochastic_depth import *

class IdentityResNextSeSd(nn.Module):
   def __init__(self, in_channels, intermediate_channels, groups=1, expansion=4, identity_downsample=None, stride=1, surv_prob=0.5):
      super().__init__()
      self.bn1 = nn.BatchNorm2d(in_channels)
      self.conv1 = nn.Conv2d(
            in_channels,
            intermediate_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
         )

      self.bn2 = nn.BatchNorm2d(intermediate_channels)
      self.conv2 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            groups=groups,
         )

      self.bn3 = nn.BatchNorm2d(intermediate_channels)
      self.conv3 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels*expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
         )

      self.activ_fn = nn.SiLU(inplace=True)
      self.identity_downsample = identity_downsample
      self.se = SqueezeExcitation(intermediate_channels*expansion, intermediate_channels)
      self.sd = StochasticDepth(p=surv_prob, shape=(1, 1, 1))

   def reset_parameters(self):
      self.bn1.reset_parameters()
      self.conv1.reset_parameters()
      self.bn2.reset_parameters()
      self.conv2.reset_parameters()
      self.bn3.reset_parameters()
      self.conv3.reset_parameters()
      self.se.reset_parameters()
      self.sd.reset_parameters()
      if (self.identity_downsample is not None):
         self.identity_downsample.reset_parameters()

   def forward(self, x):
      identity = x
      # compute the residual call
      x = self.bn1(x)
      x = self.activ_fn(x)
      x = self.conv1(x)

      x = self.bn2(x)
      x = self.activ_fn(x)
      x = self.conv2(x)

      x = self.bn3(x)
      x = self.activ_fn(x)
      x = self.conv3(x)

      x = self.se(x)
      x = self.sd(x)

      if (self.identity_downsample is not None):
         identity = self.identity_downsample(identity)

      x = x + identity  # add the identity (skip connection) and apply relu activation
      return x
