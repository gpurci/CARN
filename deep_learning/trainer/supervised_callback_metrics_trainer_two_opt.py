#!/usr/bin/python

import torch
from tqdm import tqdm
from torch import nn

from sys_function import * # este in root
sys_remove_modules("callback.callback_list")
sys_remove_modules("metrics.metrics_list")

from callback.callback_list import *
from metrics.metrics_list import *

class SupervisedCallbackMetricsTrainerTwoOpt():
   def __init__(self,
         model: nn.Module,
         optimizer: torch.optim.Optimizer,
         criterion: nn.Module,
         device:"cuda",
         callbacks:list = [],
         metrics:dict = {},
         type_compile="normal",
         disable_tqdm: bool = False, 
         transforms=None, 
         all_transforms=None, 
         lr_scheduler=None,
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
      if   (type_compile == "jit"):
         # torch.jit.script is still a very good option, often faster than torch.compile, especially on windows
         self.model = torch.jit.script(model)
      elif (type_compile == "compile"):
         # This compiles the model. See https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
         self.model.compile()
         # This compiles the step function
         self.step = torch.compile(self.step)
      # 
      self.criterion = criterion.to(self.device)  # Required for some loss functions
      # 
      self.transforms     = transforms
      self.all_transforms = all_transforms
      # 
      self.disable_tqdm = disable_tqdm
      self.best_va_acc  = 0.0
      #
      self.callbacks = CallbacksList(callbacks)
      self.metrics   = MetricsList(metrics)
      self.setOptimizer(optimizer, lr_scheduler)

   def setOptimizer(self, optimizer, lr_scheduler=None):
      self.optimizer    = optimizer
      self.lr_scheduler = lr_scheduler

   def step(self, data: torch.Tensor, target: torch.Tensor):
      predicted = self.model(data)
      loss = self.criterion(predicted, target)
      loss.backward()
      self.optimizer.step()
      self.optimizer.zero_grad()
      #
      predicted = self.model(data)
      loss = self.criterion(predicted, target)
      loss.backward()
      self.optimizer.step()
      self.optimizer.zero_grad()
      return predicted, loss

   def train(self, train_ds):
      self.model.train()

      for batch, (data, target) in enumerate(train_ds, 0):
         self.callbacks.on_train_batch_begin(batch, None)
         # We must move the data to the same device as the model
         # We can also use non_blocking=True to speed up the transfer for large tensors
         # Works when using pin_memory=True. For more details, check the references for pinning memory.
         # but this is useful only for pinned memory transfers (CPU-to-GPU)
         # In most cases, the improvement is negligible
         data   = data.to(self.device, non_blocking=True)
         target = target.to(self.device, non_blocking=True)

         if (self.transforms is not None):
            data = self.transforms(data)
         if (self.all_transforms is not None):
            data, target = self.all_transforms(data, target)
         
         predicted, loss = self.step(data, target)

         if (self.lr_scheduler is not None):
            self.lr_scheduler.step()
         # This metric is actually an approximation of an accuracy, we are checking whether the dominant class
         # predicted by the model is also equal to the dominant soft label
         # The reason we are moving the data from device back to CPU is because these calculations are usually
         # faster on CPU for small batch sizes
         # We use detach because we tell the autograd engine to not track the gradients for predicted anymore
         predicted = predicted.detach().cpu().argmax(dim=1)
         target    = target.detach().cpu().argmax(dim=1)
         loss = float(loss.item())
         logs = self.metrics(target, predicted, loss=loss)
         self.callbacks.on_train_batch_end(batch, logs)

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

      for batch, (data, target) in enumerate(val_ds, 0):
         self.callbacks.on_test_batch_begin(batch, None)
         # go to device
         data   = data.to(self.device, non_blocking=True)
         target = target.to(self.device, non_blocking=True)

         predicted = self.model(data)
         loss = self.criterion(predicted, target).item()

         # Here we don't need to argmax the target, because we have hard labels. We don't use DA during validation.
         # We don't need to detach, because we are already in inference_mode
         predicted = predicted.detach().cpu().argmax(dim=1)
         target    = target.detach().cpu()
         logs = self.metrics(target, predicted, "val_", loss=loss)
         self.callbacks.on_test_batch_end(batch, logs)

   def run(self, train_dl, val_dl, epochs: int, save_path:str):
      print(f"Running {epochs} epochs")
      for epoch in range(epochs):
         self.train(train_dl)
         train_logs = self.metrics.logs()
         print("train_logs", train_logs)
         self.callbacks.on_train_end(train_logs)
         self.val(val_dl)
         val_logs = self.metrics.logs()
         self.callbacks.on_test_end(val_logs)
         train_logs.update(val_logs)
         self.callbacks.on_epoch_end(epoch, train_logs)
      # We use tqdm to have a progress bar for the epochs. We disable inner progress bars on jupyter notebooks,
      # because either they produce a lot of output, or disable loading the notebook on GitHub.
      # If you run this script on a terminal, you can enable the inner progress bars.
      # Some more details about efficiency:
      #  * Using pin_memory=True in the DataLoader usually increases the data transfer speed from
      #    CPU RAM to GPU RAM, using pinned memory. More details in the official documentation.
      #    The downside is that pinned memory is a limited resource, and allocating too much of it can lead to
      #    system instability. Therefore, monitor your system when using pin_memory=True.

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
   def eval(self, eval_dl):
      self.model.eval()

      total = 0
      correct = 0
      total_loss = 0

      for data, target in tqdm(eval_dl, desc="Validation", leave=False, disable=self.disable_tqdm):  # Disable on notebook
         # go to device
         data   = data.to(self.device, non_blocking=True)
         target = target.to(self.device, non_blocking=True)

         predicted = self.model(data)
         loss = self.criterion(predicted, target).item()

         # Here we don't need to argmax the target, because we have hard labels. We don't use DA during validation.
         # We don't need to detach, because we are already in inference_mode
         predicted = predicted.detach().cpu().argmax(dim=1)
         target    = target.detach().cpu()
         correct += predicted.eq(target).sum().item()
         total   += data.size(0)
         total_loss += loss

      return total_loss / total, correct / total
