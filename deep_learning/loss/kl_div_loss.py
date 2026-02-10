#!/usr/bin/python

import torch
from torch import nn

class KLDivLoss(nn.Module):
   def __init__(self, **kw):
      super().__init__()
      self.kw = kw

   def forward(self, inputs, targets):
      inputs  = nn.functional.log_softmax(inputs, dim=-1)   # input
      targets = nn.functional.softmax(targets,    dim=-1)   # target
      kl_div_loss = nn.functional.kl_div(inputs, targets, **self.kw)
      return kl_div_loss
