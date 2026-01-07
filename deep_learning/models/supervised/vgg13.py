#!/usr/bin/python

from torch import nn, Tensor

class VGG13(nn.Module):
   def __init__(self, num_classes):
      super().__init__()

      self.layers = nn.Sequential(
         # Block 1
         nn.Conv2d(3, 64, kernel_size=3, padding=1),
         nn.BatchNorm2d(64),
         nn.ReLU(inplace=True),
         nn.Conv2d(64, 64, kernel_size=3, padding=1),
         nn.BatchNorm2d(64),
         nn.ReLU(inplace=True),
         nn.MaxPool2d(kernel_size=2, stride=2),

         # Block 2
         nn.Conv2d(64, 128, kernel_size=3, padding=1),
         nn.BatchNorm2d(128),
         nn.ReLU(inplace=True),
         nn.Conv2d(128, 128, kernel_size=3, padding=1),
         nn.BatchNorm2d(128),
         nn.ReLU(inplace=True),
         nn.MaxPool2d(kernel_size=2, stride=2),

         # Block 3
         nn.Conv2d(128, 256, kernel_size=3, padding=1),
         nn.BatchNorm2d(256),
         nn.ReLU(inplace=True),
         nn.Conv2d(256, 256, kernel_size=3, padding=1),
         nn.BatchNorm2d(256),
         nn.ReLU(inplace=True),
         nn.MaxPool2d(kernel_size=2, stride=2),

         # Block 4
         nn.Conv2d(256, 512, kernel_size=3, padding=1),
         nn.BatchNorm2d(512),
         nn.ReLU(inplace=True),
         nn.Conv2d(512, 512, kernel_size=3, padding=1),
         nn.BatchNorm2d(512),
         nn.ReLU(inplace=True),
         nn.MaxPool2d(kernel_size=2, stride=2),

         # Block 5
         nn.Conv2d(512, 512, kernel_size=3, padding=1),
         nn.BatchNorm2d(512),
         nn.ReLU(inplace=True),
         nn.Conv2d(512, 512, kernel_size=3, padding=1),
         nn.BatchNorm2d(512),
         nn.ReLU(inplace=True),
         nn.MaxPool2d(kernel_size=2, stride=2),

         # Classifier
         nn.Flatten(),
      )
      self.fc = nn.Linear(512, num_classes)

   def reset_parameters(self):
      self.fc.reset_parameters()
      for layer in self.layers:
         if (hasattr(layer, "reset_parameters")):
            layer.reset_parameters()

   def forward(self, x: Tensor) -> Tensor:
      x = self.layers(x)
      return self.fc(x)
