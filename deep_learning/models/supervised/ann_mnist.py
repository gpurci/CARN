#!/usr/bin/python

import torch

# PyTorch models inherit from torch.nn.Module
class MnistClassifier(nn.Module):
   def __init__(self, initializer=None):
      super(MnistClassifier, self).__init__()
      self.layers = [nn.Linear(28 * 28, 20),
                    nn.ReLU(inplace=True),
                    nn.Linear(20, train_set.getNumClass()),]
      self.block = nn.Sequential(*self.layers)
      self.initializer = initializer 

   def reset_parameters(self):
      if (self.initializer is not None):
         self.initializer(self.block.parameters())
      else:
         for layer in self.layers:
            if ((hasattr(layer, "reset_parameters")) and (callable(getattr(layer, "reset_parameters")))):
               layer.reset_parameters()

   def forward(self, x):
      x = self.block(x)
      return x
