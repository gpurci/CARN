#!/usr/bin/python

import torch
from torch import nn
from collections import OrderedDict
import warnings

from sys_function import * # este in root
sys_remove_modules("layers.inputs.att_unet_input")
sys_remove_modules("layers.unet_blocks.unet_resnet_block")

from layers.inputs.att_unet_input import *
from layers.unet_blocks.unet_resnet_block import *


class UNetResNet(nn.Module):
   def __init__(self, name, **conf):
      super().__init__()
      self.name = name
      self.input_layer = self._unpack_input_layer(**conf)
      self.unet_block  = UNetResNetBlock("resnet_unet", **conf)
      self.output_layer= self._unpack_output_layer(**conf)
      self.crop = nn.ZeroPad2d(-2)

   def _unpack_input_layer(self, **conf):
      block_conf = conf.get("input", None)
      if (block_conf is None):
         raise NameError("The config for '{}': is empty '{}'".format("input", block_conf))
      img_shape    = block_conf.get("img_shape", None)
      out_channels = block_conf.get("out_channels", None)
      frame_dim    = block_conf.get("frame_dim", None)
      num_heads    = block_conf.get("num_heads", None)
      input_layer  = AttUnetInput(img_shape, out_channels, frame_dim, num_heads)
      return input_layer

   def _unpack_output_layer(self, **conf):
      block_conf = conf.get("output", None)
      if (block_conf is None):
         raise NameError("The config for '{}': is empty '{}'".format("input", block_conf))
      in_channels  = block_conf.get("in_channels", None)
      out_channels = block_conf.get("out_channels", None)
      output_layer = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            groups=1,
         )
      return output_layer

   def reset_parameters(self):
      self.input_layer.reset_parameters()
      self.unet_block.reset_parameters()
      self.output_layer.reset_parameters()

   def forward(self, x):
      x = self.input_layer(x)
      x = self.unet_block(x)
      x = self.output_layer(x)
      x = self.crop(x)
      x = torch.tanh(x)
      return x
