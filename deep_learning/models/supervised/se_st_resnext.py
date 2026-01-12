#!/usr/bin/python

import torch
from torch import nn
import warnings

from sys_function import * # este in root
sys_remove_modules("layers.inputs.resnet_input")
sys_remove_modules("layers.resnet_blocks.se_st_identity_resnext_block")

from layers.inputs.resnet_input import *
from layers.resnet_blocks.se_st_identity_resnext_block import *

class SeStResNext(nn.Module):
   def __init__(self, name, initializer=None, **conf):
      super().__init__()
      self.name = name
      self.input  = self._unpack_input(**conf)
      self.body   = SeStIdentityResNextBlock("body", **conf)
      self.decode = SeStIdentityResNextBlock("decode", **conf)
      self.avgpool = nn.AdaptiveAvgPool2d(1)
      in_features, out_features = conf.get("output")
      self.fc      = nn.Linear(in_features, out_features)

   def _unpack_input(self, **conf):
      tmp_conf = conf.get("input")
      img_channels = tmp_conf.get("img_channels")
      out_channels = tmp_conf.get("out_channels")

      layer  = ResNetInput(img_channels, out_channels)
      return layer

   def reset_parameters(self, reset_layers=None):
      if (reset_layers is not None):
         for layer_name in reset_layers:
            if (hasattr(self, layer_name)):
               layer = getattr(self, layer_name)
               layer.reset_parameters()
            else:
               warnings.warn("\n\nW 'SeStResNext' do not has, layer name: '{}'\n\n".format(layer_name))
      else:
         self.input.reset_parameters()
         self.body.reset_parameters()
         self.decode.reset_parameters()
         self.fc.reset_parameters()

   def forward(self, x):
      x = self.input(x)
      x = self.body(x)
      x = self.decode(x)
      x = self.avgpool(x)
      x = torch.reshape(x, (x.shape[0], -1))
      x = self.fc(x)
      return x

"""
Input=(img_channels, out_channels, kernel_size, stride)
body={body_name={in_channels, expansion, stride, intermediate_channels, num_residual_blocks}}
Output=(in_features, out_features)
"""
