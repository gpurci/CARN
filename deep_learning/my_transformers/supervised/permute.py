#!/usr/bin/python

import torch
from torchvision.transforms import v2

class Permute(v2.Transform):
   def __init__(self, dims=0):
      super().__init__()
      self.dims = dims
      
   def __call__(self, x):
      return torch.permute(x,  dims=self.dims)
