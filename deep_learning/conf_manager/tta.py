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

class TTA():
   def __init__(self, model, dataset, in_shape=None):
      self.model   = model
      self.dataset = dataset
      self.in_shape= in_shape

   def selectModelType(self, device: torch.device, model_type: str):
      """This method creates a model on a device and prepares it for serving. 
      Depending on the optimization type, the model is optimized using different TorchScript jit utilities, 
      or even compiled using torch.compile.
      For a detailed description of each jit/compile method, please check the official documentation and official tutorials.
      """
      model = self.model.to(device)
      model.eval()

      if   (model_type == "raw"):
         pass
      elif (model_type == "scripted"):
         model = torch.jit.script(model)
      elif (model_type == "traced"):
         if (self.in_shape is not None):
            data  = torch.rand(self.in_shape, device=device)
            model = torch.jit.trace(model, data)
         else:
            raise RuntimeError("The 'in_shape' is None '{}'".format(self.in_shape))
      elif (model_type == "frozen"):
         model = torch.jit.freeze(torch.jit.script(model))
      elif (model_type == "optimized_for_inference"):
         model = torch.jit.optimize_for_inference(torch.jit.script(model))
      elif (model_type == "compiled"):
         if (os.name == "nt"):
            print("torch.compile is not supported on Windows. Try Linux or WSL instead.")
         else:
            model = torch.compile(model)
      else:
         raise RuntimeError("std::unreachable")
      return model

   @staticmethod
   def __noTTA(model, data):
      return model(data)

   @staticmethod
   def __mirroringTTA(model, data):
      predicted = model(data)
      predicted += model(hflip(data))
      return predicted

   @staticmethod
   def __translateTTA(model, data):
      predicted = model(data)
      #___________
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
      return predicted

   @staticmethod
   def __mirroring_and_translateTTA(model, data):
      predicted = model(data)
      #___________
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
            aux = padded[:, :, x:x + image_size, y:y + image_size]
            predicted += model(aux)
            predicted += model(hflip(aux))
      return predicted

   @staticmethod
   def __mirroringTTA(model, data):
      predicted = model(data)
      predicted += model(hflip(data))
      return predicted

   def __select_tta_inference(self, tta_type: str):
      if (tta_type == "mirroring"):
         self.__tta_inference = TTA.__mirroringTTA
      elif (tta_type == "translate"):
         self.__tta_inference = TTA.__translateTTA
      elif (tta_type == "mirroring_and_translate"):
         self.__tta_inference = TTA.__mirroring_and_translateTTA
      else:
         self.__tta_inference = TTA.__noTTA

   @timed(stdout=False, return_time=True, use_seconds=True)
   def tta_inference(self, model, batches, device: torch.device) -> float:
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
         predicted = self.__tta_inference(model, data)

         correct += predicted.cpu().argmax(dim=1).eq(target).sum().item()
         total   += data.size(0)

      return round(correct / total, 4)

   def inference(self, model, batches, device: torch.device, dtype: torch.dtype, model_type: str) -> Tuple[float, float]:
      """We use the automated mixed precision module to automatically cast to our desired data type. 
      We measure the accuracy and the elapsed time of the configuration."""

      enable_autocast = (device.type == "cuda") and (dtype != torch.float32)
      # Autocast is slow for cpu, so we disable it.
      # Also, if the device type is mps, autocast might not work (?)
      accuracy, elapsed = "N/A", "N/A"
      try:
         with torch.autocast(device_type=device.type, dtype=dtype, enabled=enable_autocast), torch.inference_mode():
            accuracy, elapsed = self.tta_inference(model, batches, device)
      except:
         # Debug only

         # import traceback
         # traceback.print_exc()
         print(f"Model type {model_type} failed on {dtype} on {device.type}")

      return accuracy, elapsed

   def __call__(self,
         model_types: Tuple[str, ...],
         dtypes: Tuple[torch.dtype, ...],
         tta_types: Tuple[str, ...],
         devices: Tuple[torch.device | None, ...],
      ):

      with tqdm(total=len(devices) * len(dtypes) * len(model_types) * len(tta_types), desc="Speed experiments") as tbar:
         for device, dtype, tta_type in product(devices, dtypes, tta_types):
            if (device is None):
               tbar.update(len(model_types))
               continue
            speed_results = PrettyTable()
            speed_results.field_names = ["Device", "Dtype", "TTA Type", "Model Type", "Accuracy", "Elapsed"]


            for model_type in model_types:
               model = self.selectModelType(device, model_type)
               self.__select_tta_inference(tta_type)
               accuracy, elapsed = self.inference(model, self.dataset, device, dtype, model_type)
               speed_results.add_row([device, dtype, tta_type, model_type, accuracy, elapsed])
               tbar.update()

            print(speed_results)
