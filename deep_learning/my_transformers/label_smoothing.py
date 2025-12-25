#!/usr/bin/python

import torch
from torchvision.transforms import v2
import torch.nn.functional as F

class LabelSmoothing(v2.Transform):
    def __init__(self, num_classes=0, smooth_size=0.09, dtype=torch.float32):
        super().__init__()
        assert (num_classes > 0), "Error: num_classes '{}' should be greater than '0'".format(num_classes)
        self.num_classes = num_classes
        self.smooth_size = smooth_size
        self.on_value = 1 - smooth_size
        self.dtype = dtype
        
    def __call__(self, x, y):
        # 
        y.requires_grad_(False)
        label_smoothing = torch.empty((y.shape[0], self.num_classes), dtype=self.dtype, requires_grad=False).to(y.device)
        label_smoothing.uniform_(0, self.smooth_size)
        # 
        y = F.one_hot(y, num_classes=self.num_classes).type(self.dtype) * self.on_value
        return x, label_smoothing+y
