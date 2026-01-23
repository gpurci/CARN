#!/usr/bin/python

from torch import nn
import numpy as np

class NumAttBlocks(nn.Module):
   def __init__(self, name, **conf):
      super().__init__()
      self.name = name
      self.block = self._unpack_block(**conf)

   def _unpack_block(self, **conf):
      block_conf = conf.get(self.name, None)
      if (block_conf is None):
         raise NameError("The config for '{}': is empty '{}'".format(self.name, block_conf))
      embed_dim  = block_conf.get("embed_dim", None)
      num_heads  = block_conf.get("num_heads", [1])
      dropouts   = block_conf.get("dropouts", [1])
      if (embed_dim is None):
         raise NameError("The config for embed_dim is empty '{}'".format(embed_dim))

      layers = []
      for num_head, dropout in zip(num_heads, dropouts):
         layer = nn.MultiheadAttention(embed_dim, num_head, dropout=dropout, bias=False, 
                                       add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, 
                                       batch_first=True, device=None, dtype=None)
         layers.append(layer)
      return nn.Sequential(*layers)

   def reset_parameters(self):
      for layer in self.block:
         if (hasattr(layer, "reset_parameters")):
            layer.reset_parameters()

   def forward(self, x):
      for layer in self.block:
         x, _ = layer(x, x, x)
      return x
