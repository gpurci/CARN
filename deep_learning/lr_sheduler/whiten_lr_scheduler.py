#!/usr/bin/python

import torch
from torch.optim.lr_scheduler import LRScheduler

class WhitenLrScheduler(LRScheduler):
   def __init__(self, optimizers, total_train_steps, whiten_bias_train_steps, last_epoch=-1):
      self.optimizers = optimizers
      self.optimizer1 = optimizers[0]
      self.optimizer2 = optimizers[1]
      self.total_train_steps = total_train_steps
      self.whiten_bias_train_steps = whiten_bias_train_steps
      self.run_step = 0
      super().__init__(optimizers[0], last_epoch)

   def reset(self):
      self.run_step = 0

   def step(self):
      for group in self.optimizer1.param_groups[:1]:
         group["lr"] = group["initial_lr"] * (1 - self.run_step / self.whiten_bias_train_steps)
      for group in self.optimizer1.param_groups[1:]+self.optimizer2.param_groups:
         group["lr"] = group["initial_lr"] * (1 - self.run_step / self.total_train_steps)
      self.run_step += 1
