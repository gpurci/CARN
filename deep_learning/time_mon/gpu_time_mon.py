#!/usr/bin/python

import pandas as pd
from pathlib import Path
import warnings

class GPUTimeMon(object):
   def __init__():
      # For accurately timing GPU code
      self.starter = torch.cuda.Event(enable_timing=True)
      self.ender = torch.cuda.Event(enable_timing=True)
      self.time_seconds = 0.0

   def start_timer(self):
      self.starter.record()

   def stop_timer(self):
      self.ender.record()
      self.torch.cuda.synchronize()
      self.time_seconds += 1e-3 * self.starter.elapsed_time(self.ender)
