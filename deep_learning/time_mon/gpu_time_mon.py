#!/usr/bin/python

import torch

class GPUTimeMon(object):
   def __init__(self):
      # For accurately timing GPU code
      self.starter = torch.cuda.Event(enable_timing=True)
      self.ender   = torch.cuda.Event(enable_timing=True)
      self.time_seconds = 0.0

   def start(self):
      self.starter.record()

   def stop(self):
      self.ender.record()
      torch.cuda.synchronize()
      self.time_seconds += 1e-3 * self.starter.elapsed_time(self.ender)

   def time(self):
      return self.time_seconds

   def reset(self):
      self.time_seconds = 0.0
