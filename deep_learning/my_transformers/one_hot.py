#!/usr/bin/python

import torch
from torchvision.transforms import v2
import torch.nn.functional as F

class OneHotTarget(v2.Transform):
    def __init__(self, num_classes=0, dtype=torch.float32):
        super().__init__()
        assert (num_classes > 0), "Error: num_classes '{}' should be greater than '0'".format(num_classes)
        self.num_classes = num_classes
        self.dtype = dtype
        
    def __call__(self, x, y):
        return x, F.one_hot(y, num_classes=self.num_classes).type(self.dtype)
