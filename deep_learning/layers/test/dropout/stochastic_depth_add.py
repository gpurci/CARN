#!/usr/bin/python

import torch
from torch import nn

class StochasticDepthAdd(nn.Module):
   def __init__(self, p=0.5, shape=()):
      super().__init__()
      assert (p >= 0), "Error: percent '{}' should be greater equal to '0'".format(p)
      assert (p <= 1), "Error: percent '{}' should be less equal to '1'".format(p)
      self.surv_prob = p
      self.shape = shape

   def reset_parameters(self):
      pass

   def forward(self, x1, x0):
      if (self.training):
         dropout_tensor = torch.rand((x0.shape[0], *self.shape), 
               device=x0.device, requires_grad=False, dtype=torch.float32) < self.surv_prob
         dropout_tensor = dropout_tensor.to(x0.dtype)
         x0 = x0 * dropout_tensor
         x1 = x1 * (1. - dropout_tensor)
         x0 = torch.div(x0, self.surv_prob)
         x1 = torch.div(x1, self.surv_prob)

      return x0+x1
