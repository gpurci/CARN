#!/usr/bin/python

import torch
from torch import nn
from collections import OrderedDict
import warnings

from sys_function import * # este in root
sys_remove_modules("layers.resnet_blocks.identity_resnet_block")
sys_remove_modules("layers.resnet_blocks.identity_resnet_up_block")
sys_remove_modules("layers.inputs.input_by_stride")

from layers.resnet_blocks.identity_resnet_block import *
from layers.resnet_blocks.identity_resnet_up_block import *
from layers.inputs.input_by_stride import *

class ResNetUnsupervised(nn.Module):
   def __init__(self, name, initializer=None, **conf):
      super().__init__()
      self.name  = name
      self.input   = self._unpack_input(**conf)
      self.encode  = IdentityResNetBlock("encode", **conf)
      self.decode  = IdentityResNetUpBlock("decode", **conf)
      in_features, out_features = conf.get("Output")
      self.fc      = nn.Conv2d(
                           in_features,
                           out_features,
                           kernel_size=1,
                           stride=1,
                           padding="same",
                           bias=False,
                           groups=1,
                        )
      self.tanh    = nn.Tanh()

   def _unpack_input(self, **conf):
      tmp_conf = conf.get("Input")
      img_channels = tmp_conf.get("img_channels")
      out_channels = tmp_conf.get("out_channels")
      kernel_size = tmp_conf.get("kernel_size")
      stride = tmp_conf.get("stride")

      layer = InputStride(img_channels, out_channels, kernel_size, stride)
      return layer

   def reset_parameters(self, reset_layers=None):
      if (reset_layers is not None):
         for layer_name in reset_layers:
            if (hasattr(self, layer_name)):
               layer = getattr(self, layer_name)
               layer.reset_parameters()
            else:
               warnings.warn("\n\nW 'ResNetUnsupervised' do not has, layer name: '{}'\n\n".format(layer_name))
      else:
         self.input.reset_parameters()
         self.encode.reset_parameters()
         self.decode.reset_parameters()
         self.fc.reset_parameters()

   def forward(self, x):
      x = self.input(x)
      x = self.encode(x)
      x = self.decode(x)
      x = self.fc(x)
      x = self.tanh(x)
      return x

"""
Input=(img_channels, out_channels, kernel_size, stride)
encode={body_name={in_channels, expansion, stride, intermediate_channels, num_residual_blocks}}
Output=(in_features, out_features)
"""
