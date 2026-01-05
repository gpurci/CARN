#!/usr/bin/python

import pandas as pd
from pathlib import Path
import warnings
from callback.callback_base import *

class SaveHistory(CallbacksBase):
   def __init__(self, filename):
      self.filename = filename
      self.pd_history = None
      self.epoch = 0
      # daca este fisierul se actualizeaza valoarea epocii
      if (isinstance(self.filename, str)):
         if (Path(self.filename).is_file()):
            try:
               self.pd_history = pd.read_csv(self.filename, on_bad_lines="skip")
               is_epoch = self.pd_history.get("epoch", None)
            except:
               self.pd_history = None
               is_epoch = None
            if (is_epoch is not None):
               self.epoch = self.pd_history.at[len(self.pd_history)-1, "epoch"]
         else:
            path = Path(self.filename).parent
            Path(path).mkdir(mode=0o777, parents=True, exist_ok=True)
            Path(self.filename).touch(mode=0o666, exist_ok=True)
      else:
         warnings.warn("\n\nCallback: Numele fisierului '{}' este type '{}'".format(self.filename, type(self.filename)))

   def on_epoch_end(self, epoch, logs=None):
      tmp_logs = logs.copy()
      tmp_logs["epoch"] = epoch+self.epoch
      for key in tmp_logs.keys():
         val = [tmp_logs[key]]
         tmp_logs[key] = val
      # salveaza logurile in data frame
      pd_df = pd.DataFrame(data=tmp_logs)
      # adauga logurile in lista de loguri
      if (self.pd_history is None):
         self.pd_history = pd_df
      else:
         self.pd_history = pd.concat([self.pd_history, pd_df], ignore_index=True)
      # salveaza in 'csv' file
      self.pd_history.to_csv(self.filename, index=False) 
