#!/usr/bin/python

from torch.utils.data import Dataset
import numpy as np

class RowDataset(Dataset):
   def __init__(self, dataset:dict, name):
      # data aquisition
      self.inputs  = dataset["inputs"]
      self.targets = dataset["targets"]
      self.name = name

   def __len__(self):
      return self.inputs.shape[0]

   def __getitem__(self, idx: int):
      inputs  = self.inputs[ idx].copy()
      outputs = self.targets[idx]
      # We clone the data here, otherwise the runtime transforms might corrupt our data. They really do!
      # You should never trust your users, even if they are yourself.
      return inputs, outputs.astype(np.int64)

   def __str__(self):
      str_info = """RowDataset:
   name: {},
   size: {},
   num_classes : {},
   shape: {}""".format(self.name, self.inputs.shape[0], self.targets.max()+1, self.inputs.shape)
      return str_info

   @staticmethod
   def help():
      info_help = """dataset: 'dict'
   inputs:  'np.array', 
   targets: 'np.array', 
   """
      print(info_help)
