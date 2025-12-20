#!/usr/bin/python

import torch
from torch.utils.data import Dataset
from torchvision import datasets
import numpy as np

class DatasetRandAppendUsupervised(Dataset):
    def __init__(self, dataset:dict, transform=None, train=True, freq_rand=10):
        # data aquisition
        self.data    = np.array(dataset["inputs"])
        self.targets = np.array(dataset["targets"])
        # data transformation
        self.transform  = transform
        # data 
        self.isTrain(train)
        # the frequency of append random (image, target)
        self.FREQ_RAND_SIZE = freq_rand
        self.count          = 0 # count for calling random (image, target)

    def __len__(self):
        return self.data.shape[0]

    def isTrain(self, train:bool):
        # data 
        if (train):
            self.apply_fn = self.__append_rand
        else:
            self.apply_fn = self.__identity

    def __append_rand(self, idx:int):
        if (self.count >= self.FREQ_RAND_SIZE):
            inputs = np.random.randint(low=0, high=255, size=self.data.shape[1:], dtype=np.uint8)
            self.count = 0
        else:
            inputs = self.data[idx].copy()
        self.count += 1
        return inputs

    def __identity(self, idx:int):
        inputs  = self.data[idx].copy()
        return inputs

    def __getitem__(self, idx: int):
        inputs = self.apply_fn(idx)
        if (self.transform is not None):
            inputs = self.transform(inputs)
        # We clone the data here, otherwise the runtime transforms might corrupt our data. They really do!
        # You should never trust your users, even if they are yourself.
        return inputs

