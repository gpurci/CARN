#!/usr/bin/python

import torch
from torchvision.transforms import v2
import torch.nn.functional as F

class Rescale(v2.Transform):
    def __init__(self, scale, offset=0.0):
        super().__init__()
        assert (scale > 0), "Error: scale '{}' should be greater than '0'".format(scale)
        self.scale  = scale
        self.offset = offset
        
    def __call__(self, x):
        return (x/self.scale)-self.offset
