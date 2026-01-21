#!/usr/bin/python

import torch
from torch import nn

from math import ceil
from prettytable import PrettyTable
from tqdm.auto import tqdm

from sys_function import * # este in root
sys_remove_modules("time_mon.gpu_time_mon")

from time_mon.gpu_time_mon import *

class FinetuneAirbatch():
   def __init__(self, model, criterion, optimizers, lr_schedulers, tta_obj, save_path, transforms=None):
      self.model = model
      self.criterion     = criterion
      self.optimizers    = optimizers
      self.lr_schedulers = lr_schedulers
      self.tta_obj       = tta_obj
      self.transforms    = transforms
      self.save_path     = save_path
      self.best_acc      = 0

   def __call__(self, train_dl, val_ds, epochs, 
            work_modes=["warmup", 0, 1],
            whiten_bias_train_epochs=3, 
         ):
      self.whiten_bias_train_epochs = whiten_bias_train_epochs

      prety_table_obj = PrettyTable()
      prety_table_obj.field_names = ["run", "epoch", "train_acc", "val_acc", "eval_acc", "time_seconds"]

      with tqdm(total=len(work_modes), desc="Speed train") as tbar:
         for work_mode in work_modes:
            train_dl.setWorkMode(work_mode)
            self.model.reset_parameters()
            for lr_sched in self.lr_schedulers:
               if (hasattr(lr_sched, "reset")):
                  lr_sched.reset()
            self.fit(train_dl, val_ds, epochs, work_mode, prety_table_obj)
            tbar.update()
      return prety_table_obj

   def save_model(self, val_acc):
      if (val_acc > self.best_acc):
         self.best_acc = val_acc
         torch.save({
            "model_state_dict": self.model.state_dict(),
         }, self.save_path)

   def fit(self, train_dl, val_ds, epochs, work_mode, prety_table_obj):
      self.time_mon_obj = GPUTimeMon()
      torch.cuda.empty_cache()
      for epoch in range(epochs):
         train_acc, time_seconds = self.train(train_dl, epoch)
         val_acc  , time_seconds = self.eval(val_ds)
         self.save_model(val_acc)
         train_acc, val_acc, time_seconds = round(train_acc, 3), round(val_acc, 3), round(time_seconds, 3)
         prety_table_obj.add_row([work_mode, epoch, train_acc, val_acc, "", time_seconds])
      eval_acc, time_seconds = self.tta(val_ds)
      eval_acc, time_seconds = round(eval_acc, 3), round(time_seconds, 3)
      prety_table_obj.add_row([work_mode, "tta", "", "", eval_acc, time_seconds])
      prety_table_obj.add_divider()

   def train(self, train_dl, epoch):
      self.model.train()
      step      = 0
      total_acc = 0
      self.time_mon_obj.start()
      for inputs, targets in train_dl:
         if (self.transforms is not None):
            inputs, targets = self.transforms(inputs, targets)
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
         step      += inputs.size(0)
      self.time_mon_obj.stop()
      return total_acc/step, self.time_mon_obj.time()

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
         step      += inputs.size(0)
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
         step      += inputs.size(0)
      self.time_mon_obj.stop()
      return total_acc/step, self.time_mon_obj.time()
