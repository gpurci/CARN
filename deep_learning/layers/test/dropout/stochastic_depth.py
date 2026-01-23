#!/usr/bin/python

import torch
from torch import nn

class StochasticDepth(nn.Module):
   def __init__(self, p=0.5, shape=()):
      super().__init__()
      assert (p >= 0), "Error: percent '{}' should be greater equal to '0'".format(p)
      assert (p <= 1), "Error: percent '{}' should be less equal to '1'".format(p)
      self.surv_prob = p
      self.shape = shape

   def reset_parameters(self):
      pass

   def forward(self, x):
      if (self.training):
         dropout_tensor = torch.rand((x.shape[0], *self.shape), 
               device=x.device, requires_grad=False, dtype=torch.float32) < self.surv_prob
         dropout_tensor = torch.div(dropout_tensor, self.surv_prob)
         x = x * dropout_tensor
      return x
