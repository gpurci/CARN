#!/usr/bin/python

import numpy as np

class RandGenerator(object):
   def __init__(self, in_size, num_classes):
      self.in_size     = in_size
      self.num_classes = num_classes
      assert (self.num_classes > 0), "Error: num_classes '{}' should be greater than '0'".format(self.num_classes)
      self.__init_rand_target(num_classes)

   def __init_rand_target(num_classes):
      if (num_classes == 1):
         self.target_gen = lambda : 0
      else:
         self.target_gen = self.__rand_target_gen

   def __rand_target_gen(self):
      return np.random.randint(low=0, high=self.num_classes, size=None, dtype=np.uint16)

   def __getitem__(self, idx: int):
      inputs  = np.random.randint(low=0, high=255, size=self.in_size, dtype=np.uint8)
      outputs = self.target_gen()
      return inputs, outputs


