#!/usr/bin/python

import torch
from torch import nn

from math import ceil
from prettytable import PrettyTable

from sys_function import * # este in root
sys_remove_modules("time_mon.gpu_time_mon")

from time_mon.gpu_time_mon import *

class FinetuneAirbatch():
   def __init__(self, model, tta_obj):
      self.model   = model
      self.time_mon_obj = GPUTimeMon()
      self.tta_obj      = tta_obj
      self.show_log_obj = PrettyTable()
      self.show_log_obj.field_names = ["run", "epoch", "train_acc", "val_acc", "eval_acc", "time_seconds"]

   def __call__(self, train_dl, val_ds, epochs, optimizers, lr_schedulers, whiten_bias_train_epochs, 
            reset_param=False, reset_scheduler=False
         ):
      self.optimizers    = optimizers
      self.lr_schedulers = lr_schedulers
      self.whiten_bias_train_epochs = whiten_bias_train_epochs

      for work_mode in ["warmup", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
         train_dl.setWorkMode(work_mode)
         if (reset_param):
            self.model.reset_parameters()
         for lr_sched in self.lr_schedulers:
            if (hasattr(lr_sched, "reset") and reset_scheduler)
               lr_sched.lr_sched()
         self.fit(train_dl, val_ds, epochs, work_mode)

   def fit(self, train_dl, val_ds, epochs, work_mode):
      for epoch in range(epochs):
         train_acc, time_seconds = self.train(train_dl, epoch)
         val_acc  , time_seconds = self.eval(val_ds)
         self.show_log_obj.add_row([work_mode, epoch, train_acc, val_acc, "", time_seconds])
      eval_acc, time_seconds = self.tta(val_ds)
      self.show_log_obj.add_row([work_mode, "", "", "", eval_acc, time_seconds])

   def train(self, train_dl, epoch):
      self.model.train()
      total_acc = 0
      self.time_mon_obj.start()
      for inputs, targets in train_dl:
         predicted = self.model(inputs, whiten_bias_grad=(epoch < self.whiten_bias_train_epochs))
         self.criterion(predicted, targets).backward()
         for lr_sched in self.lr_schedulers:
            lr_sched.step()
         for opt in self.optimizers:
            opt.step()
         self.model.zero_grad(set_to_none=True)

         # Save the accuracy and loss from the last training batch of the epoch
         predicted = predicted.detach().cpu().argmax(dim=1)
         targets   = targets.detach().cpu().argmax(dim=1)
         total_acc += predicted.eq(targets).sum().item()
      self.time_mon_obj.stop()
      return total_acc/len(train_dl), self.time_mon_obj.time()

   @torch.inference_mode()  # This is the recommended way to do this
   def eval(self, val_ds):
      self.model.eval()
      step      = 0
      total_acc = 0
      self.time_mon_obj.start()
      for inputs, targets in val_ds:
         predicted = self.model(inputs)
         # Here we don't need to argmax the targets, because we have hard labels. We don't use DA during validation.
         # We don't need to detach, because we are already in inference_mode
         predicted = predicted.detach().cpu().argmax(dim=1)
         targets   = targets.detach().cpu()
         total_acc += predicted.eq(targets).sum().item()
         step      += 1
      self.time_mon_obj.stop()
      return total_acc/step, self.time_mon_obj.time()

   @torch.inference_mode()  # This is the recommended way to do this
   def tta(self, val_ds):
      self.model.eval()
      step      = 0
      total_acc = 0
      self.time_mon_obj.start()
      for inputs, targets in val_ds:
         predicted = self.tta_obj(inputs)
         # Here we don't need to argmax the targets, because we have hard labels. We don't use DA during validation.
         # We don't need to detach, because we are already in inference_mode
         predicted = predicted.detach().cpu().argmax(dim=1)
         targets   = targets.detach().cpu()
         total_acc += predicted.eq(targets).sum().item()
         step      += 1
      self.time_mon_obj.stop()
      return total_acc/step, self.time_mon_obj.time()
