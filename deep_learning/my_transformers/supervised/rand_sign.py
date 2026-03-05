#!/usr/bin/python

import torch
from torchvision.transforms import v2
import torch.nn.functional as F

class RandSign(v2.Transform):
    def __init__(self, size_same=3):
        super().__init__()
        assert (size_same > 1), "Error: scale '{}' should be greater than '1'".format(size_same)
        self.size_same = size_same
        self.size  = 0
        self.count = 0
        self.change_sign = False
        
    def __call__(self, x):
        if (self.count >= self.size):
            self.size  = torch.randint(low=1, high=self.size_same, size=(), dtype=torch.int32).item()
            self.count = 0
            self.change_sign = self.change_sign and True
        self.count += 1
        if (self.change_sign):
            x = -x
        return x
