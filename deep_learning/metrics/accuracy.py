#!/usr/bin/python

import torch
from metrics.metrics_base import *

class Acuracy(MetricsBase):
   def __init__(self, **kw):
      super().__init__(**kw)

   def __call__(self, y, y_pred):
      # y     : true labels (batch_size,)
      # y_pred: predicted labels (batch_size, )
      correct = y_pred.eq(y).sum().item()
      total   = y.size(0)
      return correct / total
