#!/usr/bin/python

import torch
from torchvision.transforms import v2

class Sum(v2.Transform):
   def __init__(self, dim=0):
      super().__init__()
      self.dim = dim
      
   def __call__(self, x):
      return torch.sum(x,  dim=self.dim)
