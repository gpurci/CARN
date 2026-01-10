#!/usr/bin/python

import torch
from torch.utils.data import Dataset
from torchvision import datasets
import numpy as np

class CallDatasetAppend(Dataset):
   def __init__(self, dataset:dict, append_datasets=None, transform=None, train=True, percent_from=0.1):
      # data aquisition
      self.__init_base_dataset(dataset)
      self.__init_virtual_dataset(append_datasets)
      self.__init_virtual_size(percent_from)
      # data transformation
      self.transform = transform
      # data 
      self.isTrain(train)

   def __init_base_dataset(self, dataset):
      # data aquisition
      self.dataset   = dataset["dataset"]
      self.task_size = dataset["size"]
      self.task_num_classes = dataset["num_classes"]
      assert (self.task_num_classes > 0), "Error: num_classes '{}' should be greater than '0'".format(self.task_num_classes)

   def __init_virtual_dataset(self, append_datasets):
      if (append_datasets is not None):
         inputs  = [self.inputs]
         targets = [self.targets]
         self.num_classes = self.task_num_classes
         for dataset in append_datasets:
            inputs.append( dataset["inputs"])
            targets.append(dataset["targets"]+self.num_classes)
            self.num_classes += dataset["num_classes"]
         else:
            self.inputs  = np.concatenate(inputs,  axis=0)
            self.targets = np.concatenate(targets, axis=0)

   def __init_virtual_size(self, percent_from):
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
         self.apply_fn = self.__append
         self.size = self.task_size + self.SIZE_VIRTUAL_DATA
      else:
         self.apply_fn = self.__identity
         self.size = self.task_size

   def __append(self, idx:int):
      if (idx >= self.task_size):
         append_idx = np.random.randint(low=self.task_size, high=self.inputs.shape[0], size=None, dtype=np.int32)
         inputs  = self.inputs[ append_idx].copy()
         outputs = self.targets[append_idx]
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
