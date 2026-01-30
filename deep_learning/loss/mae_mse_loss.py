#!/usr/bin/python

import torch
from torch import nn

class MAEMSELoss(nn.Module):
   def __init__(self, alpha=0.5):
      super().__init__()
      self.alpha = nn.Parameter(torch.tensor(alpha))

   def forward(self, pred, target):
      l1 = torch.mean(torch.abs(pred - target))
      l2 = torch.mean((pred - target) ** 2)
      return self.alpha * l1 + (1 - self.alpha) * l2
