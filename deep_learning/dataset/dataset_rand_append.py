#!/usr/bin/python

import torch
from torch.utils.data import Dataset
from torchvision import datasets
import numpy as np

class DatasetRandAppend(Dataset):
    def __init__(self, dataset:dict, transform=None, train=True, rand_class_size=4, freq_rand=10):
        # data aquisition
        self.data    = dataset["inputs"]
        self.targets = dataset["targets"]
        self.dataset_num_classes = dataset["num_classes"]
        # data transformation
        self.transform  = transform
        # data 
        self.isTrain(train)
        # dataset number classes
        assert (self.dataset_num_classes > 0), "Error: num_classes '{}' should be greater than '0'".format(self.dataset_num_classes)
        # total number of targets with append target
        self.num_classes = self.dataset_num_classes + rand_class_size
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
            inputs  = np.random.randint(low=0, high=255, size=self.data.shape[1:], dtype=np.uint8)
            outputs = np.random.randint(low=self.dataset_num_classes, high=self.num_classes, size=None,
                                        dtype=np.uint16)
            self.count = 0
        else:
            inputs  = self.data[idx].copy()
            outputs = self.targets[idx]
        self.count += 1
        return inputs, outputs

    def __identity(self, idx:int):
        inputs  = self.data[idx].copy()
        outputs = self.targets[idx]
        return inputs, outputs

    def __getitem__(self, idx: int):
        inputs, outputs = self.apply_fn(idx)
        if (self.transform is not None):
            inputs = self.transform(inputs)
        # We clone the data here, otherwise the runtime transforms might corrupt our data. They really do!
        # You should never trust your users, even if they are yourself.
        return inputs, outputs.astype(np.int64)
