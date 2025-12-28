#!/usr/bin/python

import torch
from torch import nn
from collections import OrderedDict

from sys_function import * # este in root
sys_remove_modules("layers.resnet.identity_resnext_se_sd")
sys_remove_modules("layers.resnet.identity_conv2d_downsample")

from layers.resnet.identity_resnext_se_sd import *
from layers.resnet.identity_conv2d_downsample import *

class IdentityResNextSeSdBlock(nn.Module):
   def __init__(self, name, **conf):
      super().__init__()
      self.name = name
      self.block = self._unpack_block(**conf)

   def _unpack_block(self, **conf):
      body_conf = conf.get(self.name)
      layers = []
      for key in body_conf.keys():
         layer_conf   = body_conf.get(key)
         block_layers = self._unpack_blocks(key, **layer_conf)
         layers.extend(block_layers)
      return nn.Sequential(OrderedDict(layers))

   def _unpack_blocks(self, key, **conf):
      in_channels = conf.get("in_channels", None)
      expansion   = conf.get("expansion", 4)
      stride      = conf.get("stride", 1)
      groups      = conf.get("groups", 1)
      surv_prob   = conf.get("surv_prob", 0.5)
      intermediate_channels = conf.get("intermediate_channels", None)
      num_residual_blocks   = conf.get("num_residual_blocks", 1)
      out_channels = intermediate_channels * expansion
      #print("name {}, in_channels {}, out_channels {}, expansion {}, stride {}, intermediate_channels {}, num_residual_blocks {}".format(key, in_channels, out_channels, expansion, stride, intermediate_channels, num_residual_blocks))
      layers = []
      for idx in range(num_residual_blocks):
         identity_downsample = None
         if ((stride != 1) or (in_channels != out_channels)):
            identity_downsample = IdentityConv2dDownSample(in_channels, out_channels, stride=stride)
         layer = IdentityResNextSeSd(in_channels, intermediate_channels, 
                                 groups=groups,
                                 expansion=expansion, identity_downsample=identity_downsample, 
                                 stride=stride, surv_prob=surv_prob)
         in_channels = out_channels
         stride = 1
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
name={block_name={in_channels, groups, expansion, stride, intermediate_channels, num_residual_blocks, surv_prob}}
"""
