#!/usr/bin/python

import torch
from torch import nn

class KLDivLoss(nn.Module):
   def __init__(self, **kw):
      super().__init__()
      self.kw = kw

   def forward(self, pred, target):
      kl_div_loss = nn.functional.kl_div(pred, target, **self.kw)
      return kl_div_loss - 1
