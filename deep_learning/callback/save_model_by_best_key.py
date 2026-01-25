#!/usr/bin/python

import torch
from pathlib import Path
import numpy as np
from callback.callback_base import *

class SaveModelByBestKey(CallbacksBase):
   def __init__(self, model, filename, key="val_acc"):
      """"""
      self.setModel(model)
      self.filename = filename
      self.key      = key
      self.val_best_key = 0
      path = Path(filename).parent
      Path(path).mkdir(mode=0o777, parents=True, exist_ok=True)
      Path(filename).touch(mode=0o666, exist_ok=True)

   def on_epoch_end(self, epoch, logs=None):
      if ((self.key in logs) and (logs[self.key] > self.val_best_key)):
         self.val_best_key = logs[self.key]
         torch.save({"model_state_dict": self.model.state_dict()}, self.filename)
