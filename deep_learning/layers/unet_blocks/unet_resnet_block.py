#!/usr/bin/python

import torch
from torch import nn
from collections import OrderedDict

from sys_function import * # este in root
sys_remove_modules("layers.resnet_blocks.identity_resnet_block")
sys_remove_modules("layers.unet_blocks.identity_resnet_up_block")

from layers.resnet_blocks.identity_resnet_block import *
from layers.unet_blocks.identity_resnet_up_block import *

class UNetResNetBlock(nn.Module):
   def __init__(self, name, **conf):
      super().__init__()
      self.name  = name
      self.encode = IdentityResNetBlock("encode", **conf)
      self.decode = IdentityResNetUpBlock("decode", **conf)

   def reset_parameters(self):
      self.encode.reset_parameters()
      self.decode.reset_parameters()

   def forward(self, x):
      enc_features = []
      enc_features.append(x)
      for layer in self.encode.block:
         x = layer(x)
         enc_features.append(x)
      for layer, x_feature in zip(self.decode.block, enc_features[::-1]):
         x = layer(x)
         x = x + x_feature
      return x

"""
name={block_name={in_channels, expansion, stride, intermediate_channels, num_residual_blocks}}
"""
