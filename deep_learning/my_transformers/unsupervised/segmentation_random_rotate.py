#!/usr/bin/python

import torch
from torchvision.transforms import v2
import numpy as np

class SegmentationRandomRotation(v2.Transform):
   def __init__(self, degrees, fill=0):
      super().__init__()
      self.degrees = degrees
      self.fill    = fill

   def __call__(self, image, mask):
      angle = np.random.uniform(-self.degrees, self.degrees)

      image = v2.functional.rotate(
         image,
         angle,
         interpolation=v2.functional.InterpolationMode.BILINEAR,
         fill=self.fill
      )

      mask = v2.functional.rotate(
         mask,
         angle,
         interpolation=v2.functional.InterpolationMode.NEAREST,
         fill=self.fill
      )

      return image, mask