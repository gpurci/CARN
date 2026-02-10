#!/usr/bin/python

import torch
from torch import nn

class MAEMSELoss(nn.Module):
   def __init__(self, alpha=0.5, reduction="mean"):
      super().__init__()
      self.alpha = nn.Parameter(torch.tensor(alpha))
      self.reduction = reduction

   def forward(self, pred, target):
      l1 = torch.abs(pred - target)
      l2 = (pred - target) ** 2

      if   (self.reduction == "mean"):
         l1 = torch.mean(l1)
         l2 = torch.mean(l2)
      elif (self.reduction == "sum"):
         l1 = torch.sum(l1)
         l2 = torch.sum(l2)
      elif (self.reduction == "none"):
         pass

      return self.alpha * l1 + (1 - self.alpha) * l2
