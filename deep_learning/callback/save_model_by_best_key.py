#!/usr/bin/python

import torch
from pathlib import Path
import numpy as np
from callback.callback_base import *

class SaveModelByBestKey(CallbacksBase):
   def __init__(self, model, filename, key="val_acc", mode="min"):
      """"""
      self.setModel(model)
      self.filename = filename
      self.key      = key
      self.mode     = mode
      # 
      self.best = np.inf if mode == "min" else -np.inf
      # make path folder if does not exist
      path = Path(filename).parent
      Path(path).mkdir(mode=0o777, parents=True, exist_ok=True)
      Path(filename).touch(mode=0o666, exist_ok=True)

   def _is_improvement(self, current):
      if (self.mode == "min"):
         b_val = current < self.best
      else:
         b_val = current > self.best
      return b_val

   def on_epoch_end(self, epoch, logs=None):
      if ((self.key in logs) and (self._is_improvement(logs[self.key]))):
         self.best = logs[self.key]
         torch.save({"model_state_dict": self._model.state_dict()}, self.filename)
