#!/usr/bin/python

import torch
from torch.utils.data import Dataset
from torchvision import datasets

class MnistDataset(Dataset):
    def __init__(self, in_transform, out_transform, class_size=4, train=True):
        # data aquisition
        dataset  = datasets.MNIST(root='./data', train=train, transform=None, download=True)
        self.data    = dataset.data.numpy()
        self.targets = dataset.targets.numpy()
        # data transformation
        self.in_transform  = in_transform
        self.out_transform = out_transform
        # MNIST dataset number classes + append class
        self.class_size = class_size

    def getNumClass(self):
        return self.class_size
    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, i: int):
        inputs  = self.data[i]
        outputs = self.targets[i]
        if (self.in_transform is not None):
            inputs = self.in_transform(inputs)
        if (self.out_transform is not None):
            outputs = self.out_transform(outputs)
        return (inputs, outputs)