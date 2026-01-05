#!/usr/bin/python

import torch
from pathlib import Path
import numpy as np
from callback.callback_base import *

class SaveBestAccVal(CallbacksBase):
   def __init__(self, model, filename):
      """"""
      self.model    = model
      self.filename = filename
      self.best_acc = 0
      path = Path(filename).parent
      Path(path).mkdir(mode=0o777, parents=True, exist_ok=True)
      Path(filename).touch(mode=0o666, exist_ok=True)

   def on_epoch_end(self, epoch, logs=None):
      if (("val_acc" in logs) and (logs["val_acc"] > self.best_acc)):
         self.best_acc = logs["val_acc"]
         torch.save({"model_state_dict": self.model.state_dict()}, self.filename)
