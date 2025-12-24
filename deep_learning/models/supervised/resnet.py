#!/usr/bin/python

import torch
from torch import nn
import warnings

from sys_function import * # este in root
sys_remove_modules("layers.resnet_blocks.identity_resnet_block")
sys_remove_modules("layers.resnet_blocks.input_by_stride")

from layers.resnet_blocks.identity_resnet_block import *
from layers.resnet_blocks.input_by_stride import *

class ResNet(nn.Module):
   def __init__(self, name, initializer=None, **conf):
      super().__init__()
      self.name = name
      self.input = self._unpack_input(**conf)
      self.body  = IdentityResNetBlock("body", **conf)
      self.decode = IdentityResNetBlock("decode", **conf)
      self.avgpool = nn.AdaptiveAvgPool2d(1)
      in_features, out_features = conf.get("Output")
      self.fc      = nn.Linear(in_features, out_features)
      self.softmax = nn.Softmax(dim=0)

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
               warnings.warn("\n\nW 'ResNet' do not has, layer name: '{}'\n\n".format(layer_name))
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
