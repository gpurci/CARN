#!/usr/bin/python

import torch
from torch.utils.data import Dataset
from torchvision import datasets

class MnistDatasetRandAppend(Dataset):
    def __init__(self, in_transform, out_transform, rand_class_size=4, freq_rand=10):
        # data aquisition
        dataset  = datasets.MNIST(root='./data', train=True, transform=None, download=True)
        self.data    = dataset.data.numpy()
        self.targets = dataset.targets.numpy()
        # data transformation
        self.in_transform  = in_transform
        self.out_transform = out_transform
        # data 
        self.rand_in   = torch.empty(self.data.shape[1:])
        # MNIST dataset number classes
        self.mnist_target_size = 10
        # total number of targets with append target
        self.target_size = self.mnist_target_size + rand_class_size
        # the frequency of append random (image, target)
        self.FREQ_RAND_SIZE = freq_rand
        self.count          = 0 # count for calling random (image, target)

    def getNumClass(self):
        return self.target_size
    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, i: int):
        if (self.count >= self.FREQ_RAND_SIZE):
            inputs  = self.rand_in.uniform_(-1.0, 1.0).numpy()
            outputs = torch.randint(low=self.mnist_target_size, high=self.target_size, size=(1,)).item()
            self.count = 0
        else:
            inputs  = self.data[i]
            outputs = self.targets[i]
        self.count += 1
        if (self.in_transform is not None):
            inputs = self.in_transform(inputs)
        if (self.out_transform is not None):
            outputs = self.out_transform(outputs)
        return (inputs, outputs)
