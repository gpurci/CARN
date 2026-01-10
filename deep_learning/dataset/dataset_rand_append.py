#!/usr/bin/python

import torch
from torch.utils.data import Dataset
from torchvision import datasets
import numpy as np

class DatasetRandAppend(Dataset):
   def __init__(self, dataset:dict, transform=None, train=True, virtual_classes=4, percent_from=0.1):
      # data aquisition
      self.__init_base_dataset(dataset)
      self.__init_virtual_size(percent_from)
      # data transformation
      self.transform  = transform
      # total number of targets with append target
      self.num_classes = self.task_num_classes + virtual_classes
      # data 
      self.isTrain(train)

   def __init_base_dataset(self, dataset):
     # data aquisition
     self.inputs  = dataset["inputs"]
     self.targets = dataset["targets"]
     self.task_num_classes = dataset["num_classes"]
     assert (self.task_num_classes > 0), "Error: num_classes '{}' should be greater than '0'".format(self.task_num_classes)
     self.task_size = self.inputs.shape[0]

   def __init_virtual_size(self, percent_from)
      # dataset number classes
      assert (percent_from >= 0), "Error: percent_from '{}' should be greater equal to '0'".format(percent_from)
      assert (percent_from <= 1), "Error: percent_from '{}' should be less equal to '1'".format(percent_from)
      # the frequency of append random (image, target)
      self.SIZE_VIRTUAL_DATA = int(self.task_size*percent_from)

   def __len__(self):
      return self.size

   def isTrain(self, train:bool):
      # data 
      if (train):
         self.apply_fn = self.__append_rand
         self.size = self.task_size + self.SIZE_VIRTUAL_DATA
      else:
         self.apply_fn = self.__identity
         self.size = self.task_size

   def __append_rand(self, idx:int):
      if (idx >= self.task_size):
         inputs  = np.random.randint(low=0, high=255, size=self.inputs.shape[1:], dtype=np.uint8)
         outputs = np.random.randint(low=self.task_num_classes, high=self.num_classes, size=None,
                              dtype=np.uint16)
      else:
         inputs  = self.inputs[ idx].copy()
         outputs = self.targets[idx]
      return inputs, outputs

   def __identity(self, idx:int):
      inputs  = self.inputs[ idx].copy()
      outputs = self.targets[idx]
      return inputs, outputs

   def __getitem__(self, idx: int):
      inputs, outputs = self.apply_fn(idx)
      if (self.transform is not None):
         inputs = self.transform(inputs)
      # We clone the data here, otherwise the runtime transforms might corrupt our data. They really do!
      # You should never trust your users, even if they are yourself.
      return inputs, outputs.astype(np.int64)
