#!/usr/bin/python

from torch.utils.data import Dataset
import numpy as np

class AddVirtualDatasets(Dataset):
   def __init__(self, meta_ds:dict, help_metads=None, transform=None, help_from=0.1):
      # data aquisition
      self.__init_main_dataset(meta_ds)
      self.__init_help_datasets(help_metads)
      self.__init_virtual_size(help_metads, help_from)
      # data transformation
      self.transform = transform
      # data 
      train = help_metads is not None
      self.isTrain(train)
      self.count = 0

   def __init_main_dataset(self, meta_ds):
      # data aquisition
      self.datasets = [meta_ds["data_reader"]]
      self.main_ds_size = meta_ds["size"]
      self.num_classes  = meta_ds["num_classes"]
      assert (self.num_classes > 0), "Error: task num_classes '{}' should be greater than '0'".format(self.num_classes)
      self.help_ds_size  = [self.main_ds_size]
      self.ds_name = ["main"]

   def __init_help_datasets(self, help_metads):
      if (help_metads is not None):
         for key in help_metads.keys():
            meta_ds = help_metads[key]
            self.datasets.append(meta_ds["data_reader"])
            self.help_ds_size.append(meta_ds["size"])
            self.ds_name.append(key)
      # 
      self.help_ds_size  = np.array(self.help_ds_size)

   def __init_virtual_size(self, help_metads, help_from):
      if (help_metads is not None):
         # percent from dataset size for append real image
         assert (help_from >= 0), "Error: help_from '{}' should be greater equal to '0'".format(help_from)
         assert (help_from <= 1), "Error: help_from '{}' should be less equal to '1'".format(help_from)
         # the frequency of append random (image, target)
         self.VIRTUAL_SIZE  = int(self.main_ds_size*help_from)
         size_help_ds = len(help_metads.keys())
         self.SIZE_PER_HELP = self.VIRTUAL_SIZE // size_help_ds
      else:
         self.VIRTUAL_SIZE  = 0
         self.SIZE_PER_HELP = 0

   def __len__(self):
      return self.size

   def __str__(self):
      str_info = """AddVirtualDatasets:
   size: {},
   num_classes : {},
   virtual size: {},
   size per help dataset: {},
   """.format(self.size, self.num_classes, self.VIRTUAL_SIZE, self.SIZE_PER_HELP)
      # 
      for data_reader, ds_size, name in zip(self.datasets, self.help_ds_size, self.ds_name):
         str_info += """
name: {},
   data_reader: {}, 
   dataset size: {},
   """.format(name, data_reader, ds_size)
      return str_info

   def isTrain(self, train:bool):
      # data 
      if (train):
         self.apply_fn = self.__append
         self.size = self.main_ds_size + self.VIRTUAL_SIZE
      else:
         self.apply_fn = self.__identity
         self.size = self.main_ds_size

   def __who_get_ds(self, idx):
      idx_run = idx - self.main_ds_size
      if (idx_run >= 0):
         idx_ds    = (idx_run // self.SIZE_PER_HELP) + 1
         idx_input = np.random.randint(low=0, high=self.help_ds_size[idx_ds], size=None, dtype=np.int32)
      else:
         idx_ds, idx_input = 0, idx
      return idx_ds, idx_input

   def __append(self, idx:int):
      idx_ds, idx_input = self.__who_get_ds(idx)
      inputs, targets = self.datasets[idx_ds][idx_input]
      inputs = inputs.copy()
      if (idx_ds > 0):
         targets = self.count
         self.count = (self.count + 1) % self.num_classes
      return inputs, targets

   def __identity(self, idx:int):
      inputs, targets = self.datasets[0][idx]
      inputs = inputs.copy()
      return inputs, targets

   def __getitem__(self, idx: int):
      inputs, targets = self.apply_fn(idx)
      if (self.transform is not None):
         inputs = self.transform(inputs)
      # We clone the data here, otherwise the runtime transforms might corrupt our data. They really do!
      # You should never trust your users, even if they are yourself.
      return inputs, np.int64(targets)

   @staticmethod
   def help():
      info_help = """meta_ds: 'dict', main dataset
   data_reader - 'object' that has '__getitem__' method return tuple (input, target)
   size        - 'int' size of all dataset
   num_classes - 'int' number of classes
help_metads: 'dict' of meta_ds, where key is name of dataset and value is meta_ds
transform  : 'tranforms' or function to transform input
help_from  : 'float' percent from size of main dataset, equal distributed for all help dataset
   ex: size main dataset = 100, help_from = 0.1; total size of help dataset is 110, 10 is distributed for all help dataset
   """
      print(info_help)
