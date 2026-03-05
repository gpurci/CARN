#!/usr/bin/python

import torch
from torchvision.transforms import v2
import numpy as np

class SegmentationCropResize(v2.Transform):
   def __init__(self, size, p=None):
      super().__init__()
      assert (size is not None), "Error: size '{}' should be 'int' or 'tuple'".format(size)
      self.size  = size
      if (p is not None):
         self.p = [p, 1-p]
      else:
         self.p = p
      
   def __call__(self, inputs, targets):
      # 
      inputs.requires_grad_(False)
      targets.requires_grad_(False)
      #
      select = np.random.choice(2, size=None, p=self.p)
      if (select == 0):
         # Resize
         inputs  = v2.functional.resize(inputs,  self.size, interpolation=v2.functional.InterpolationMode.BILINEAR)
         targets = v2.functional.resize(targets, self.size, interpolation=v2.functional.InterpolationMode.NEAREST)
      else:
         input_shape = inputs.shape[1:]
         if ((input_shape[0] < self.size[0]) or (input_shape[1] < self.size[1])):
            # Resize
            inputs  = v2.functional.resize(inputs,  self.size, interpolation=v2.functional.InterpolationMode.BILINEAR)
            targets = v2.functional.resize(targets, self.size, interpolation=v2.functional.InterpolationMode.NEAREST)


      # Random crop parameters
      i, j, h, w = v2.RandomCrop.get_params(inputs, self.size)
      #
      inputs  = v2.functional.crop(inputs,  i, j, h, w)
      targets = v2.functional.crop(targets, i, j, h, w)

      return inputs, targets
