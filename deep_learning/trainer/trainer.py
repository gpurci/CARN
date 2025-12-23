#!/usr/bin/python

import torch
from tqdm import tqdm
from torch import nn

class Trainer():
   def __init__(self,
         model: nn.Module,
         optimizer: torch.optim.Optimizer,
         criterion: nn.Module,
         device="cuda",
         type_compile="normal",
         disable_tqdm: bool = False, 
         transforms=None, 
         all_transforms=None, 
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
      self.optimizer = optimizer
      # 
      self.transforms     = transforms
      self.all_transforms = all_transforms
      # 
      self.disable_tqdm = disable_tqdm
      self.best_va_acc  = 0.0

   def setOptimizer(self, optimizer):
      self.optimizer = optimizer

   def step(self, data: torch.Tensor, target: torch.Tensor):
      predicted = self.model(data)
      loss = self.criterion(predicted, target)
      loss.backward()
      self.optimizer.step()
      self.optimizer.zero_grad()
      return predicted, loss

   def train(self, train_ds):
      self.model.train()

      total = 0
      correct = 0
      total_loss = 0

      for data, target in tqdm(train_ds, desc="Training", leave=False, disable=self.disable_tqdm):  # Disable on notebook
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

         if target.ndim > 1:
            # We do this when cutmix or mixup was used, transforming the hard labels into soft labels
            target = target.argmax(1)
         # This metric is actually an approximation of an accuracy, we are checking whether the dominant class
         # predicted by the model is also equal to the dominant soft label
         # The reason we are moving the data from device back to CPU is because these calculations are usually
         # faster on CPU for small batch sizes
         # We use detach because we tell the autograd engine to not track the gradients for predicted anymore
         correct += predicted.detach().cpu().argmax(dim=1).eq(target.detach().cpu()).sum().item()
         total   += data.size(0)
         total_loss += float(loss.item())

      return total_loss / total, correct / total

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

      total = 0
      correct = 0
      total_loss = 0

      for data, target in tqdm(val_ds, desc="Validation", leave=False, disable=self.disable_tqdm):  # Disable on notebook
         # go to device
         data   = data.to(self.device, non_blocking=True)
         target = target.to(self.device, non_blocking=True)

         predicted = self.model(data)
         loss = self.criterion(predicted, target).item()

         # Here we don't need to argmax the target, because we have hard labels. We don't use DA during validation.
         # We don't need to detach, because we are already in inference_mode
         correct += predicted.detach().cpu().argmax(dim=1).eq(target.detach().cpu()).sum().item()
         total   += data.size(0)
         total_loss += loss

      return total_loss / total, correct / total

   def run(self, train_dl, val_dl, epochs: int, save_path:str):
      print(f"Running {epochs} epochs")
      with tqdm(range(epochs), desc="Training") as pbar:
         for _ in pbar:
            tr_loss, tr_acc = self.train(train_dl)
            va_loss, va_acc = self.val(val_dl)
            print({"tr_loss":tr_loss, "tr_acc":tr_acc, "va_loss":va_loss, "va_acc":va_acc})
            if va_acc > self.best_va_acc:
               self.best_va_acc = va_acc
               torch.save({
                  'model_state_dict': self.model.state_dict(),
                  'model_name': self.model.name,
               }, save_path)
            pbar.set_postfix(
                           train_loss=float(tr_loss),
                           train_acc=float(tr_acc),
                           val_loss=float(va_loss),
                           val_acc=float(va_acc),)
      return {"tr_loss":tr_loss, "tr_acc":tr_acc, "va_loss":va_loss, "va_acc":va_acc}
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
         correct += predicted.detach().cpu().argmax(dim=1).eq(target.detach().cpu()).sum().item()
         total   += data.size(0)
         total_loss += loss

      return total_loss / total, correct / total
