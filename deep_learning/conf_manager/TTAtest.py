#!/usr/bin/python

import os
import torch
from torch import nn

from torchvision.transforms import v2
from torchvision.transforms.v2.functional import hflip

import warnings
from itertools import product
from typing import Tuple
from prettytable import PrettyTable

from timed_decorator.simple_timed import timed
from tqdm import tqdm


from sys_function import * # este in root
sys_remove_modules("callback.callback")

#from callback.callback import *

class RunTTA():
   def __init__(self, model, trainer, epochs, train_dl, val_dl, history_path, reset_layers=None):
      self.model   = model
      self.trainer = trainer
      self.epochs  = epochs
      self.train_dl = train_dl
      self.val_dl   = val_dl
      self.callback = Callback(filename=history_path, freq=1)
      self.reset_layers = reset_layers


   def create_model(self, model, device: torch.device, model_type: str, in_shape=None):
      """This method creates a model on a device and prepares it for serving. 
      Depending on the optimization type, the model is optimized using different TorchScript jit utilities, 
      or even compiled using torch.compile.
      For a detailed description of each jit/compile method, please check the official documentation and official tutorials.
      """
      model = model.to(device)
      model.eval()

      if   (model_type == "raw"):
         pass
      elif (model_type == "scripted"):
         model = torch.jit.script(model)
      if (model_type == "traced"):
         if (in_shape is not None):
            data = torch.rand(in_shape, device=self.device)
            model = torch.jit.trace(model, data)
         else:
            raise RuntimeError("The 'in_shape' is None '{}'".format(in_shape))

      if (model_type == "frozen"):
         return torch.jit.freeze(torch.jit.script(model))
      if (model_type == "optimized_for_inference"):
         return torch.jit.optimize_for_inference(torch.jit.script(model))
      if (model_type == "compiled"):
         if (os.name == "nt"):
            print("torch.compile is not supported on Windows. Try Linux or WSL instead.")
            return model
         return torch.compile(model)
      else:
         raise RuntimeError("std::unreachable")
      return model

@timed(stdout=False, return_time=True, use_seconds=True)
def tta_inference(model, batches: Tuple[Tuple[torch.Tensor, torch.Tensor], ...], device: torch.device,
              tta_type: str) -> float:
   """This function performs inference and TTA, while measuring the elapsed time.

   There are 4 versions of TTA:

      no TTA
      mirroring: performs a horizontal flip for the input images, doing an additional inference pass
      translate: performs 8 translations of the input images, doing 8 additional inference passes
      mirroring_and_translate

   """
   total = 0
   correct = 0

   for data, target in batches:
      data = data.to(device)

      predicted = model(data)
      if tta_type == "mirroring":
         predicted += model(hflip(data))
      elif tta_type == "translate":
         padding_size = 2
         image_size = 32
         # We pad using the same value the model has seen during training
         padded = v2.functional.pad(data, [padding_size], fill=0.5)
         for i in [-2, 0, 2]:
            for j in [-2, 0, 2]:
               if i == 0 and j == 0:
                  continue
               x = padding_size + i
               y = padding_size + j
               predicted += model(padded[:, :, x:x + image_size, y:y + image_size])
      elif tta_type == "mirroring_and_translate":
         padding_size = 2
         image_size = 32
         padded = v2.functional.pad(data, [padding_size], fill=0.5)
         for i in [-2, 0, 2]:
            for j in [-2, 0, 2]:
               if i == 0 and j == 0:
                  continue
               x = padding_size + i
               y = padding_size + j
               aux = padded[:, :, x:x + image_size, y:y + image_size]
               predicted += model(aux)
               predicted += model(hflip(aux))

      correct += (predicted.cpu().argmax(dim=1) == target).sum().item()
      total += data.size(0)

   return round(correct / total, 4)

def inference(model, batches: Tuple[Tuple[torch.Tensor, torch.Tensor], ...], device: torch.device, tta_type: str,
           dtype: torch.dtype, model_type: str) -> Tuple[float, float]:

   """We use the automated mixed precision module to automatically cast to our desired data type. 
   We measure the accuracy and the elapsed time of the configuration."""

   enable_autocast = device.type == "cuda" and dtype != torch.float32
   # Autocast is slow for cpu, so we disable it.
   # Also, if the device type is mps, autocast might not work (?)
   accuracy, elapsed = "N/A", "N/A"
   try:

      with torch.autocast(device_type=device.type, dtype=dtype, enabled=enable_autocast), torch.inference_mode():
         accuracy, elapsed = tta_inference(model, batches, device, tta_type)
   except:
      # Debug only

      # import traceback
      # traceback.print_exc()
      print(f"Model type {model_type} failed on {dtype} on {device.type}")

   return accuracy, elapsed

def do_speed_test(data: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
              model_types: Tuple[str, ...],
              dtypes: Tuple[torch.dtype, ...],
              tta_types: Tuple[str, ...],
              devices: Tuple[torch.device | None, ...],
              model_path: str):
   tta_type = "none"
   with tqdm(total=len(devices) * len(dtypes) * len(model_types), desc="Speed experiments") as tbar:
      for device, dtype in product(devices, dtypes):
         if device is None:
            tbar.update(len(model_types))
            continue
         speed_results = PrettyTable()
         speed_results.field_names = ["Device", "Dtype", "TTA Type", "Model Type", "Accuracy", "Elapsed"]

         for model_type in model_types:
            model = create_model(model_path, device, model_type)
            accuracy, elapsed = inference(model, data, device, tta_type, dtype, model_type)
            speed_results.add_row([device, dtype, tta_type, model_type, accuracy, elapsed])
            tbar.update()

         print(speed_results)

def do_tta_test(data: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
            model_types: Tuple[str, ...],
            dtypes: Tuple[torch.dtype, ...],
            tta_types: Tuple[str, ...],
            devices: Tuple[torch.device | None, ...],
            model_path: str):
   tta_results = PrettyTable()
   tta_results.field_names = ["Device", "Dtype", "TTA Type", "Model Type", "Accuracy", "Elapsed"]

   device = devices[0] if devices[0] is not None else devices[1]
   model_type = "scripted model"

   for dtype, tta_type in tqdm(tuple(product(dtypes, tta_types)), desc="TTA experiments"):
      if device is None:
         continue
      model = create_model(model_path, device, model_type)
      accuracy, elapsed = inference(model, data, device, tta_type, dtype, model_type)
      tta_results.add_row([device, dtype, tta_type, model_type, accuracy, elapsed])

   print(tta_results)

