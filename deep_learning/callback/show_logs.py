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
      self.max_logs = None
      self.min_logs = None

   def on_epoch_begin(self, epoch, logs=None):
      pass

   def on_epoch_end(self, epoch, logs=None):
      clear_output(wait=True)
      plt.close("all")
      self.__update_logs(logs)
      self.show()

   def _save(self):
      if (self.filename is not None):
         plt.savefig(self.filename)

   def _show(self):
      plt.show()
      plt.close("all")

   def show(self):
      keys = self.__get_unique_logs()
      figsize = int(round(len(keys)/2, 0))
      self.__put(keys, figsize)
      info = self._get_statistic()
      self._save()
      self._show()
      print(info)

   def _get_statistic(self):
      info = ""
      for key in self.logs.keys():
         act = self.logs[key][-1]
         min_key = self.min_logs[key]
         max_key = self.max_logs[key]
         info += "{}:\n\tmin: {},\tmax: {},\tcur: {}\n".format(key, min_key, max_key, act)

      return info

   def setLogs(self, logs):
      self.logs = logs
      if (logs is not None):
         self.min_logs = {}
         self.max_logs = {}
         for key in logs.keys():
            self.min_logs[key] = logs[key][0]
            self.max_logs[key] = logs[key][0]
         # update last
         for key in logs.keys():
            for i in range(len(logs[key])):
               self.update_statistic(self.min_logs, key, logs[key][i], mode="min")
               self.update_statistic(self.max_logs, key, logs[key][i], mode="max")

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
               self.update_statistic(self.min_logs, key, logs[key], mode="min")
               self.update_statistic(self.max_logs, key, logs[key], mode="max")
         else:
            self.logs = {}
            self.min_logs = {}
            self.max_logs = {}
            for key in logs.keys():
               self.logs[key] = [logs[key]]
               self.max_logs[key] = logs[key]
               self.min_logs[key] = logs[key]

   def update_statistic(self, prev_logs, key, val, mode):
      if (mode == "min"):
         if (prev_logs[key] >= val):
            prev_logs[key] = round(val, 5)
      else:
         if (prev_logs[key] <= val):
            prev_logs[key] = round(val, 5)

   def __put(self, keys, figsize):
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
