#!/usr/bin/python

import copy
import torch

import warnings
from itertools import product
from typing import Tuple
from prettytable import PrettyTable
from prettytable import TableStyle

from timed_decorator.simple_timed import timed
from tqdm.auto import tqdm

from sys_function import * # este in root
sys_remove_modules("time_mon.gpu_time_mon")
sys_remove_modules("metrics.metrics_list")

from time_mon.gpu_time_mon import *
from metrics.metrics_list import *

class FinetuneTTAClassification():
   def __init__(self, dataset, tta_obj, select_model, metrics:dict = {}):
      self.dataset = dataset
      self.tta_obj = tta_obj
      self.select_model = select_model
      self.time_mon_obj = GPUTimeMon()
      self.metrics = MetricsList(metrics)

   def tta_inference(self, device: torch.device) -> float:
      """This function performs inference and TTA, while measuring the elapsed time.
      """
      total   = 0
      correct = 0

      torch.cuda.empty_cache()
      self.time_mon_obj.reset()
      self.time_mon_obj.start()
      for inputs, targets in self.dataset:
         inputs  = inputs.to(device)
         targets = targets.to(device).detach().cpu()
         predicted = self.tta_obj(inputs).detach().cpu()
         # 
         correct += predicted.argmax(dim=1).eq(targets).sum().item()
         total   += inputs.size(0)
      self.time_mon_obj.stop()

      return round(correct / total, 4), round(self.time_mon_obj.time(), 4)

   def inference(self, device: torch.device, dtype: torch.dtype, model_type: str) -> Tuple[float, float]:
      """We use the automated mixed precision module to automatically cast to our desired data type. 
      We measure the accuracy and the elapsed time of the configuration."""

      enable_autocast = (device.type == "cuda") and (dtype != torch.float32)
      # Autocast is slow for cpu, so we disable it.
      # Also, if the device type is mps, autocast might not work (?)
      accuracy, elapsed = "N/A", "N/A"
      try:
         with torch.autocast(device_type=device.type, dtype=dtype, enabled=enable_autocast), torch.inference_mode():
            accuracy, elapsed = self.tta_inference(device)
      except RuntimeError as e:
         # Debug only
         print(f"Error: '{e}', model type '{model_type}' failed on '{dtype}' on '{device.type}'")

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
            prety_table_obj = PrettyTable()
            prety_table_obj.field_names = ["Device", "Dtype", "TTA Type", "Model Type", "Accuracy", "Elapsed"]


            for model_type in model_types:
               model = self.selectModelType(device, model_type)
               self.tta_obj.model = model
               self.tta_obj.select(tta_type)
               accuracy, elapsed = self.inference(device, dtype, model_type)
               prety_table_obj.add_row([device, dtype, tta_type, model_type, accuracy, elapsed])
               tbar.update()

            print(prety_table_obj)

   def by(self,
         model_types: Tuple[str, ...],
         dtypes: Tuple[torch.dtype, ...],
         tta_types: Tuple[str, ...],
         devices: Tuple[torch.device | None, ...],
         ord_prod:dict,
         table_style=TableStyle.MARKDOWN,
      ):

      list_args = [model_types, dtypes, tta_types, devices]
      list_keys = ["model_type", "dtype", "tta_type", "device"]
      args = list_args.copy()
      for i, key in enumerate(list_keys, 0):
         args[ord_prod[key]] = list_args[i]
      prevs = [args[i][0] for i in range(len(args)-1)]

      prety_table_obj = PrettyTable()
      prety_table_obj.field_names = ["Device", "Dtype", "TTA Type", "Model Type", "Accuracy", "Elapsed"]
      prety_table_obj.set_style(table_style)

      with tqdm(total=len(devices) * len(dtypes) * len(model_types) * len(tta_types), desc="Speed experiments") as tbar:
         for act_args in product(*args):

            for i in range(len(prevs)):
               prev = prevs[i]
               if (prev != act_args[i]):
                  prevs[i] = act_args[i]
                  prety_table_obj.add_divider()

            device, dtype = act_args[ord_prod["device"]], act_args[ord_prod["dtype"]]
            tta_type, model_type = act_args[ord_prod["tta_type"]], act_args[ord_prod["model_type"]]

            if (device is None):
               tbar.update(len(model_types))
               continue
               
            model = self.select_model(device, model_type)
            model.eval()
            self.tta_obj.model = model
            self.tta_obj.select(tta_type)
            accuracy, elapsed = self.inference(device, dtype, model_type)
            prety_table_obj.add_row([device, dtype, tta_type, model_type, accuracy, elapsed])
            tbar.update()

         print(prety_table_obj)
      return prety_table_obj

   # @torch.no_grad()  # This is what you usually see in tutorials
   @torch.inference_mode()  # This is the recommended way to do this
   def eval(self, device, model_type):
      model = self.select_model(device, model_type)
      model.eval()

      lst_predicts = []
      lst_targets  = []

      for inputs, targets in tqdm(self.dataset, desc="Validation", leave=False):  # Disable on notebook
         # go to device
         inputs  = inputs.to(device,  non_blocking=True)
         targets = targets.to(device, non_blocking=True)

         predicted = model(inputs)
         # Here we don't need to argmax the targets, because we have hard labels. We don't use DA during validation.
         # We don't need to detach, because we are already in inference_mode
         predicted = predicted.detach().cpu().argmax(dim=1)
         targets   = targets.detach().cpu()
         # 
         lst_predicts.append(predicted)
         lst_targets.append(targets)
         

      return torch.cat(lst_targets, 0), torch.cat(lst_predicts, 0)
