#!/usr/bin/python

import torch
from metrics.metrics_base import *

class F1Score(MetricsBase):
   def __init__(self, num_classses, epsilon=1e-7, **kw):
      super().__init__(**kw)
      self.num_classses = torch.arange(num_classses)
      self.epsilon      = epsilon

   def __call__(self, y, y_pred):
      tp = 0
      fp = 0
      fn = 0

      for c in self.num_classses:
         tp += torch.sum(((y == c) & (y_pred == c)).float())
         fp += torch.sum(((y != c) & (y_pred == c)).float())
         fn += torch.sum(((y == c) & (y_pred != c)).float())

      #
      precision = tp / (tp + fp + self.epsilon)
      recall    = tp / (tp + fn + self.epsilon)
      f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
      return f1.item()
