#!/usr/bin/python

import os
import torch
from torch import nn
import warnings

from sys_function import * # este in root
sys_remove_modules("callback.callback_list")
sys_remove_modules("metrics.metrics_list")

from callback.callback_list import *
from metrics.metrics_list import *

class UnsupervisedClbkMTrainer():
   def __init__(self,
         model: nn.Module,
         optimizers: torch.optim.Optimizer,
         criterion: nn.Module,
         device:"cuda",
         callbacks:list = [],
         metrics:dict = {},
         model_type="raw",
         inputs_transforms=None, 
         target_transforms=None, 
         lr_schedulers=None,
   ):
      self.device = device
      print(f"Using device: {self.device}")
      # Efficiency stuff
      if (self.device.type == "cuda"):
         # This flag tells pytorch to use the cudnn auto-tuner to find the most efficient convolution algorithm for
         # This training.
         torch.backends.cudnn.benchmark = True
         # Check this: https://docs.pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
         torch.set_float32_matmul_precision("high")

      # We don't need to shuffle the validation set
      self.model = model.to(self.device)  # The model must be on the same device
      self.selectModelType(model_type)
      # 
      self.criterion = criterion.to(self.device)  # Required for some loss functions
      # 
      self.inputs_transforms = inputs_transforms
      self.target_transforms = target_transforms
      #
      self.callbacks = CallbacksList(callbacks)
      self.metrics   = MetricsList(metrics)
      self.setOptimizers(optimizers, lr_schedulers)

   def selectModelType(self, model_type):
      if   (model_type == "raw"):
         pass
      elif (model_type == "scripted"):
         # torch.jit.script is still a very good option, often faster than torch.compile, especially on windows
         self.model = torch.jit.script(self.model)
      elif (model_type == "compiled"):
         if (os.name == "nt"):
            warnings.warn("\n\ntorch.compile is not supported on Windows. Try Linux or WSL instead.\n\n")
         else: # TO DO: check if is linux
            # This compiles the model. See https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
            self.model = torch.compile(self.model)
      else:
         raise RuntimeError("std::unreachable")

   def setOptimizers(self, optimizers, lr_schedulers=None):
      self.optimizers    = optimizers
      self.lr_schedulers = lr_schedulers

   def train(self, train_ds):
      self.model.train()

      for batch, (inputs, targets) in enumerate(train_ds, 0):
         #self.callbacks.on_train_batch_begin(batch, None)
         # We must move the dataset to the same device as the model
         # We can also use non_blocking=True to speed up the transfer for large tensors
         # Works when using pin_memory=True. For more details, check the references for pinning memory.
         # but this is useful only for pinned memory transfers (CPU-to-GPU)
         # In most cases, the improvement is negligible
         inputs  = inputs.to(self.device,  non_blocking=True)
         targets = targets.to(self.device, non_blocking=True)

         if (self.inputs_transforms is not None):
            inputs, targets = self.inputs_transforms(inputs, targets)
         if (self.target_transforms is not None):
            targets = self.target_transforms(inputs)

         predicted = self.model(inputs)
         loss = self.criterion(predicted, targets)
         loss.backward()

         for opt in self.optimizers:
            opt.step()
            opt.zero_grad()
         if (self.lr_schedulers is not None):
            for sched in self.lr_schedulers:
               sched.step()
         self.model.zero_grad(set_to_none=True)

         # This metric is actually an approximation of an accuracy, we are checking whether the dominant class
         # predicted by the model is also equal to the dominant soft label
         # The reason we are moving the dataset from device back to CPU is because these calculations are usually
         # faster on CPU for small batch sizes
         # We use detach because we tell the autograd engine to not track the gradients for predicted anymore
         predicted = predicted.detach().cpu()
         targets   = targets.detach().cpu()
         loss = float(loss.mean().detach().cpu().item())
         logs = self.metrics(targets, predicted, loss=loss)
         #self.callbacks.on_train_batch_end(batch, logs)

   @torch.inference_mode()  # This is the recommended way to do this
   def val(self, val_ds):
      self.model.eval()

      for batch, (inputs, targets) in enumerate(val_ds, 0):
         # go to device
         inputs  = inputs.to(self.device,  non_blocking=True)
         targets = targets.to(self.device, non_blocking=True)
         if (self.target_transforms is not None):
            targets = self.target_transforms(inputs)

         predicted = self.model(inputs)
         loss = self.criterion(predicted, targets)

         # Here we don't need to argmax the targets, because we have hard labels. We don't use DA during validation.
         # We don't need to detach, because we are already in inference_mode
         predicted = predicted.detach().cpu()
         targets   = targets.detach().cpu()
         loss = float(loss.mean().detach().cpu().item())
         logs = self.metrics(targets, predicted, "val_", loss=loss)

   def run(self, train_dl, val_dl, epochs: int):
      print(f"Running {epochs} epochs")
      for epoch in range(epochs):
         self.callbacks.on_epoch_begin(epoch, None)
         torch.cuda.empty_cache()
         self.train(train_dl)
         train_logs = self.metrics.logs()
         self.val(val_dl)
         val_logs   = self.metrics.logs()
         train_logs.update(val_logs)
         self.callbacks.on_train_end(train_logs)
         try:
            self.callbacks.on_epoch_end(epoch, train_logs)
         except Exception as e:
            print(e)
            break
         #print("train_logs", train_logs)
