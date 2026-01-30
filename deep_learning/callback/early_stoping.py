#!/usr/bin/python

import torch
import math
from callback.callback_base import *

class EarlyStoping(CallbacksBase):
   def __init__(
      self,
      monitor="val_loss",
      mode="min",
      patience=10,
      min_delta=0.0,
      verbose=True,
   ):
      self.monitor = monitor
      self.mode = mode
      self.patience = patience
      self.min_delta = min_delta
      self.verbose = verbose

      self.best = math.inf if mode == "min" else -math.inf
      self.wait = 0
      self.best_state = None

      if (mode not in ("min", "max")):
         raise ValueError("mode must be 'min' or 'max'")

   def _is_improvement(self, current):
      if (self.mode == "min"):
         b_val = current < (self.best - self.min_delta)
      else:
         b_val = current > (self.best + self.min_delta)
      return b_val

   def on_epoch_end(self, epoch, logs):
      current = logs.get(self.monitor, None)

      if (current is None):
         raise KeyError("Metric '{}' not found in logs".format(self.monitor))

      # Convert tensors safely
      if (torch.is_tensor(current)):
         current = current.item()

      if (self._is_improvement(current)):
         self.best = current
         self.wait = 0

         if (self.verbose):
            print(
               f"[EarlyStopping] Epoch {epoch}: "
               f"{self.monitor} improved to {current:.6f}"
            )
      else:
         self.wait += 1
         if (self.verbose):
            print(
               f"[EarlyStopping] Epoch {epoch}: "
               f"no improvement ({self.wait}/{self.patience})"
            )

         if (self.wait >= self.patience):
            raise Exception("EarlyStoping")
