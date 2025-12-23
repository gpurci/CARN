#!/usr/bin/python

import torch
from torch import nn
from collections import OrderedDict

from sys_function import * # este in root
sys_remove_modules("layers.resnet_blocks.identity_resnet_module")
sys_remove_modules("layers.resnet_blocks.input_by_stride")
sys_remove_modules("layers.identity_conv2d_downsample")
sys_remove_modules("layers.conv2d_downsample")

from layers.resnet_blocks.identity_resnet_module import *
from layers.resnet_blocks.input_by_stride import *
from layers.identity_conv2d_downsample import *
from layers.conv2d_downsample import *

class ResNet(nn.Module):
   def __init__(self, name, initializer=None, **conf):
      super().__init__()
      self.name = name
      self.input = self._unpack_input(**conf)
      self.body  = self._unpack_body(**conf)
      self.decode = self._unpack_decode(**conf)
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

   def _unpack_body(self, **conf):
      body_conf = conf.get("body")
      layers = []
      for key in body_conf.keys():
         layer_conf   = body_conf.get(key)
         block_layers = self._unpack_blocks(key, **layer_conf)
         layers.extend(block_layers)
      return nn.Sequential(OrderedDict(layers))

   def _unpack_decode(self, **conf):
      body_conf = conf.get("decode")
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
      intermediate_channels = conf.get("intermediate_channels", None)
      num_residual_blocks   = conf.get("num_residual_blocks", 1)
      out_channels = intermediate_channels * expansion
      #print("name {}, in_channels {}, out_channels {}, expansion {}, stride {}, intermediate_channels {}, num_residual_blocks {}".format(key, in_channels, out_channels, expansion, stride, intermediate_channels, num_residual_blocks))
      layers = []
      for idx in range(num_residual_blocks):
         identity_downsample = None
         if ((stride != 1) or (in_channels != out_channels)):
            identity_downsample = IdentityConv2dDownSample(in_channels, out_channels, stride=stride)
         layer = IdentityResNetModule(in_channels, intermediate_channels, 
                                 expansion=expansion, identity_downsample=identity_downsample, stride=stride)
         in_channels = out_channels
         stride = 1
         block_name = "{}_{}".format(key, idx)
         layers.append((block_name, layer))
      return layers

   def reset_parameters(self, reset_layers=None):
      if (reset_layers is not None):
         for layer_name in reset_layers:
            if (hasattr(self, layer_name)):
               layer = getattr(self, layer_name)
               if (isinstance(layer, torch.nn.modules.container.Sequential)):
                  seq_layers = layer
                  for layer in seq_layers:
                     layer.reset_parameters()
               else:
                  layer.reset_parameters()
      else:
         self.input.reset_parameters()
         self.fc.reset_parameters()
         for layer in self.body:
            layer.reset_parameters()
         for layer in self.decode:
            layer.reset_parameters()

   def forward(self, x):
      x = self.input(x)
      x = self.body(x)
      x = self.decode(x)
      x = self.avgpool(x)
      x = torch.reshape(x, (x.shape[0], -1))
      x = self.fc(x)
      #x = self.softmax(x)
      return x

"""
Input=(img_channels, out_channels, kernel_size, stride)
body={body_name={in_channels, expansion, stride, intermediate_channels, num_residual_blocks}}
Output=(in_features, out_features)
"""
