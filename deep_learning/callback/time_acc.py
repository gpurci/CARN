#!/usr/bin/python

import time
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from callback.callback_base import *


class TimeAccuracy(CallbacksBase):
   def __init__(self):
      self.start = None
      self.epoch = []
      self.dt  = []
      self.acc = []

   def time(self):
      return time.time()

   def convert(self, seconds):
      return float(time.strftime("%H.%M", time.gmtime(seconds)))
   
   def on_epoch_begin(self, epoch, logs=None):
      if (self.start is None):
         self.start = self.time()

   def on_epoch_end(self, epoch, logs=None):
      dt = self.time() - self.start  # delta-time
      
      self.epoch.append(epoch)
      self.acc.append(logs["val_acc"])
      self.dt.append(self.convert(dt))

      fig, axs = plt.subplots(2, 1)
      # set the spacing between subplots
      plt.subplots_adjust(left=0.1,
                     bottom=0.1,
                     right=0.9,
                     top=0.9,
                     wspace=0.09,
                     hspace=0.0009)

      # set the spacing between subplots
      fig.suptitle("{}".format("Time monitoring"))

      axs[0].plot(self.epoch, self.acc, "-")
      axs[0].set_ylabel("Accuracy")
      axs[0].set_xlabel("Number of epoch")

      axs[1].plot(self.epoch, self.dt, "-")
      axs[1].set_ylabel("Time (hour:min)")
      axs[1].set_xlabel("Number of epoch")

      plt.show()
