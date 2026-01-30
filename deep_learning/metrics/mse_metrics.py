#!/usr/bin/python

import torch
from torch import nn
from metrics.metrics_base import *

class MSEMetrics(MetricsBase):
   def __init__(self, **kw):
      super().__init__(**kw)

   def __call__(self, y, y_pred):
      # y     : true labels (batch_size,)
      # y_pred: predicted labels (batch_size, )
      mse = nn.functional.mse_loss(y, y_pred, size_average=None, reduce=None, reduction="mean", weight=None)
      return mse
