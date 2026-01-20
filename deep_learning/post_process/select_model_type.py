#!/usr/bin/python

import os
import torch
import warnings

class SelectModelType():
   def __init__(self, model_path, build_model, in_shape=None):
      self.model_path  = model_path
      self.build_model = build_model
      self.in_shape    = in_shape

   def __call__(self, device: torch.device, model_type: str):
      """This method creates a model on a device and prepares it for serving. 
      Depending on the optimization type, the model is optimized using different TorchScript jit utilities, 
      or even compiled using torch.compile.
      For a detailed description of each jit/compile method, please check the official documentation and official tutorials.
      """
      model_data = torch.load(self.model_path, weights_only=True)

      model = self.build_model()
      model = model.to(device)
      model.load_state_dict(model_data["model_state_dict"])
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
