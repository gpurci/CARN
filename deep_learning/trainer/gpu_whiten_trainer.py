#!/usr/bin/python

import os
import torch
from tqdm.auto import tqdm
from torch import nn
import warnings

from sys_function import * # este in root
sys_remove_modules("metrics.metrics_list")

from metrics.metrics_list import *

class GPUWhitenTrainer():
   def __init__(self,
            model: nn.Module,
            optimizers: torch.optim.Optimizer,
            criterion: nn.Module,
            device:"cuda",
            metrics:dict = {},
            model_type="raw",
            disable_tqdm: bool = False, 
            transforms=None, 
            all_transforms=None, 
            lr_schedulers=None,
            whiten_bias_train_epochs=5,
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
      self.transforms     = transforms
      self.all_transforms = all_transforms
      # 
      self.disable_tqdm = disable_tqdm
      self.best_va_acc  = 0.0
      # 
      self.metrics = MetricsList(metrics)
      self.setOptimizers(optimizers, lr_schedulers)
      #
      self.whiten_bias_train_epochs = whiten_bias_train_epochs

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
      self.optimizer1    = optimizers[0]
      self.optimizer2    = optimizers[1]
      self.lr_schedulers = lr_schedulers

   def train(self, train_ds, epoch):
      self.model.train()

      for batch, (inputs, targets) in enumerate(train_ds, 0):

         if (self.transforms is not None):
            inputs = self.transforms(inputs)
         if (self.all_transforms is not None):
            inputs, targets = self.all_transforms(inputs, targets)
         
         predicted = self.model(inputs, whiten_bias_grad=(epoch < self.whiten_bias_train_epochs))
         loss = self.criterion(predicted, targets)
         loss.backward()

         for obj in self.lr_schedulers:
            obj.step()
         for opt in self.optimizers:
            opt.step()
         self.model.zero_grad(set_to_none=True)

         # This metric is actually an approximation of an accuracy, we are checking whether the dominant class
         # predicted by the model is also equal to the dominant soft label
         # The reason we are moving the dataset from device back to CPU is because these calculations are usually
         # faster on CPU for small batch sizes
         # We use detach because we tell the autograd engine to not track the gradients for predicted anymore
         predicted = predicted.detach().cpu().argmax(dim=1)
         targets   = targets.detach().cpu().argmax(dim=1)
         loss = float(loss.item())
         logs = self.metrics(targets, predicted, loss=loss)

   # Here we use the inference_mode. We are telling pytorch we are doing just inference, we don't need to track
   # tensor operations with the Autograd engine for automatic differentiation. This is also what torch.no_grad() does.
   # torch.inference_mode() = torch.no_grad() + promising torch we will never use any tensor created in this scope in
   # autograd tracked operations.
   # This promise allows additional optimizations, such as removing version tracking from tensors. If we violate the
   # promise, and use a tensor created in the inference_mode scope in an operation for which we need to calculate the
   # gradient, we should expect errors.
   # Recapitulating:
   #  * If we will never use Autograd, inference_mode is more optimized.
   #  * If we use Autograd, but just don't want to track some operations using Autograd, use no_grad.
   # @torch.no_grad()  # This is what you usually see in tutorials
   @torch.inference_mode()  # This is the recommended way to do this
   def val(self, val_ds):
      self.model.eval()

      for batch, (inputs, targets) in enumerate(val_ds, 0):

         predicted = self.model(inputs, whiten_bias_grad=False)
         loss = self.criterion(predicted, targets).item()

         # Here we don't need to argmax the targets, because we have hard labels. We don't use DA during validation.
         # We don't need to detach, because we are already in inference_mode
         predicted = predicted.detach().cpu().argmax(dim=1)
         targets   = targets.detach().cpu()
         logs = self.metrics(targets, predicted, "val_", loss=loss)

   def run(self, train_dl, val_dl, epochs: int, save_path:str):
      print(f"Running {epochs} epochs")
      with tqdm(range(epochs), desc="Training") as pbar:
         for epoch in pbar:
            torch.cuda.empty_cache()
            self.train(train_dl, epoch)
            train_logs = self.metrics.logs()
            self.val(val_dl)
            val_logs = self.metrics.logs()
            train_logs.update(val_logs)
            if (val_logs["val_acc"] > self.best_va_acc):
               self.best_va_acc = val_logs["val_acc"]
               torch.save({
                  "model_state_dict": self.model.state_dict(),
               }, save_path)
            pbar.set_postfix(**train_logs)
      return train_logs
      # We use tqdm to have a progress bar for the epochs. We disable inner progress bars on jupyter notebooks,
      # because either they produce a lot of output, or disable loading the notebook on GitHub.
      # If you run this script on a terminal, you can enable the inner progress bars.
      # Some more details about efficiency:
      #  * Using pin_memory=True in the DataLoader usually increases the dataset transfer speed from
      #    CPU RAM to GPU RAM, using pinned memory. More details in the official documentation.
      #    The downside is that pinned memory is a limited resource, and allocating too much of it can lead to
      #    system instability. Therefore, monitor your system when using pin_memory=True.

   @torch.inference_mode()  # This is the recommended way to do this
   def evaluate(self, val_dl):
      with tqdm(range(len(val_dl)), desc="Evaluate") as pbar:
         for batch in pbar:
            torch.cuda.empty_cache()
            self.val(val_dl)
            val_logs = self.metrics.logs()
            pbar.set_postfix(**val_logs)
      return val_logs
