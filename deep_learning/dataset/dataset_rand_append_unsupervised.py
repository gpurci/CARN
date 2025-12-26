#!/usr/bin/python

import torch
from torch.utils.data import Dataset
from torchvision import datasets
import numpy as np

class DatasetRandAppendUsupervised(Dataset):
    def __init__(self, dataset:dict, transform=None, train=True, percent_from=0.1):
        # data aquisition
        self.data    = np.array(dataset["inputs"])
        self.targets = np.array(dataset["targets"])
        # data transformation
        self.transform  = transform
        assert (percent_from >= 0), "Error: percent_from '{}' should be greater equal to '0'".format(percent_from)
        assert (percent_from <= 1), "Error: percent_from '{}' should be less equal to '1'".format(percent_from)
        # the frequency of append random (image, )
        self.SIZE_VIRTUAL_DATA = int(self.data.shape[0]*percent_from)
        # data 
        self.isTrain(train)

    def __len__(self):
        return self.data.shape[0]

    def isTrain(self, train:bool):
        # data 
        if (train):
            self.apply_fn = self.__append_rand
            self.size = self.data.shape[0] + self.SIZE_VIRTUAL_DATA
        else:
            self.apply_fn = self.__identity
            self.size = self.data.shape[0]

    def __append_rand(self, idx:int):
        if (idx >= self.data.shape[0]):
            inputs = np.random.randint(low=0, high=255, size=self.data.shape[1:], dtype=np.uint8)
        else:
            inputs = self.data[idx].copy()
        return inputs

    def __identity(self, idx:int):
        inputs = self.data[idx].copy()
        return inputs

    def __getitem__(self, idx: int):
        inputs = self.apply_fn(idx)
        if (self.transform is not None):
            inputs = self.transform(inputs)
        # We clone the data here, otherwise the runtime transforms might corrupt our data. They really do!
        # You should never trust your users, even if they are yourself.
        return inputs

