#!/usr/bin/python

import os
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output
from callback.callback_base import *


class ShowLogs(CallbacksBase):
   def __init__(self, title, filename=None):
      self.title    = title
      self.filename = filename
      self.logs     = None

   def on_epoch_begin(self, epoch, logs=None):
      pass

   def on_epoch_end(self, epoch, logs=None):
      clear_output(wait=True)
      plt.close("all")
      self.__update_logs(logs)
      keys = self.__get_unique_logs()
      figsize = int(round(len(keys)/2, 0))
      self.__show(keys, figsize)

   def __get_unique_logs(self):
      keys = list()
      for key in self.logs.keys():
         if ("val_" in key):
            key = key[len("val_"):]
         keys.append(key)
      keys = np.unique(keys)
      return keys

   def __update_logs(self, logs):
      if (logs is not None):
         if (self.logs is not None):
            for key in logs.keys():
               self.logs[key].append(logs[key])
         else:
            self.logs = {}
            for key in logs.keys():
               self.logs[key] = [logs[key]]

   def __show(self, keys, figsize):
      fig = plt.figure(figsize=(10, 5*figsize))
      fig.suptitle(self.title)
      fig.subplots_adjust(
               left=0.1,
               right=0.95,
               top=0.9,
               bottom=0.1,
               wspace=0.3,
               hspace=0.3
            )

      for idx in range(len(keys)):
         key = keys[idx]
         ax = fig.add_subplot(figsize, 2, idx+1)
         ax.set_title(key)
         ax.plot(self.logs[key], label="Training")
         if ("val_"+key in self.logs.keys()):
            ax.plot(self.logs["val_"+key], label="Validation")
         ax.legend()

      fig.show()
      if (self.filename is not None):
         plt.savefig(self.filename)
