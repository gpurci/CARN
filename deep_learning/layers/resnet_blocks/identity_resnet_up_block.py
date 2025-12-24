#!/usr/bin/python

import torch
from torch import nn
from collections import OrderedDict

from sys_function import * # este in root
sys_remove_modules("layers.resnet_blocks.identity_resnet_module")
sys_remove_modules("layers.identity_conv2d_downsample")
sys_remove_modules("layers.conv2d_upsample")

from layers.resnet_blocks.identity_resnet_module import *
from layers.identity_conv2d_downsample import *
from layers.conv2d_upsample import *

class IdentityResNetUpBlock(nn.Module):
   def __init__(self, name, **conf):
      super().__init__()
      self.name  = name
      self.block = self._unpack_block(**conf)

   def _unpack_block(self, **conf):
      block_conf = conf.get(self.name)
      layers = []
      for key in block_conf.keys():
         layer_conf   = block_conf.get(key)
         block_layers = self._unpack_blocks(key, **layer_conf)
         layers.extend(block_layers)
      return nn.Sequential(OrderedDict(layers))

   def _unpack_blocks(self, key, **conf):
      in_channels = conf.get("in_channels", None)
      expansion   = conf.get("expansion", 4)
      stride      = conf.get("stride", 1)
      intermediate_channels = conf.get("intermediate_channels", None)
      num_residual_blocks   = conf.get("num_residual_blocks", 1)
      out_channels = intermediate_channels * expansion
      #print("name {}, in_channels {}, out_channels {}, expansion {}, stride {}, intermediate_channels {}, num_residual_blocks {}".format(key, in_channels, out_channels, expansion, stride, intermediate_channels, num_residual_blocks))
      layers = []
      for idx in range(num_residual_blocks):
         identity_downsample = None
         if (stride != 1):
            layer  = Conv2dUpSample(in_channels, kernel_size=3, stride=stride)
            stride = 1
            block_name = "{}_{}_upsample".format(key, idx)
            layers.append((block_name, layer))
         if (in_channels != out_channels):
            identity_downsample = IdentityConv2dDownSample(in_channels, out_channels, stride=stride)
         layer = IdentityResNetModule(in_channels, intermediate_channels, 
                                 expansion=expansion, identity_downsample=identity_downsample, stride=1)
         in_channels = out_channels
         block_name = "{}_{}".format(key, idx)
         layers.append((block_name, layer))
      return layers

   def reset_parameters(self):
      for layer in self.block:
         layer.reset_parameters()

   def forward(self, x):
      x = self.block(x)
      return x

"""
name={block_name={in_channels, expansion, stride, intermediate_channels, num_residual_blocks}}
"""
