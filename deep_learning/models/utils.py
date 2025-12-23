#!/usr/bin/python

import torch

def copy_parameters(dst, src):
   with torch.no_grad():
      for param_dst, param_src in zip(dst.parameters(), src.parameters()):
         param_dst.copy_(param_src)

def freeze_layer(layer):
   for param in layer.parameters():
      param.requires_grad = False

def is_layer_frozen(layer):
   return all(not p.requires_grad for p in layer.parameters())
