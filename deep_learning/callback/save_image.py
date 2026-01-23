#!/usr/bin/python

import torch
from torch import nn

from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path

from callback.callback_base import *


# Define a callback for printing the learning rate at the end of each epoch.
class SaveGrayImg(CallbacksBase):
   def __init__(self, dataset, transform, size=1, path=""):
      self.dataset   = dataset
      self.transform = transform
      self.figsize = (size, 2) # (rows, column)
      self.path    = path
      
      Path(self.path).mkdir(mode=0o777, parents=True, exist_ok=True)
      filename = "{}/ep_{}.png".format(self.path, "init")
      self.__init_fig("init")
      self.predict()
      self.save(filename)

   def setModel(self, model):
      self.__model = model

   def __init_fig(self, epoch):
      # save a image using extension
      fig = plt.figure(figsize=self.figsize)
      fig.set_figheight(5)
      fig.set_figwidth(5)
      # set the spacing between subplots
      fig.suptitle("This is epoch {}".format(epoch), fontsize=16)

      fig.supxlabel("Generated class")
      fig.supylabel("Different generation")

      plt.subplots_adjust(left=0.1,
                          bottom=0.1,
                          right=0.9,
                          top=0.9,
                          wspace=0.09,
                          hspace=0.0009)

   def _predict(self):
      images = []
      for idx in range(self.figsize[0]):
         inputs, _ = self.dataset[idx]
         if (not hasattr(self, "__model")):
            try:
               gray = torch.unsqueeze(inputs, dim=0)
               gray = self.__model(gray)
               gray = torch.squeeze(gray, dim=0)
               gray = self.transform(gray)
               gray = torch.permute(gray, dims=(1, 2, 0)).numpy().astype(np.uint8)
            except Exception as e:
               gray = np.zeros((10, 10, 1), dtype=np.uint8)
               print("Error prediction, false image: '{}'".format(e))
         else:
            gray = np.zeros((10, 10, 1), dtype=np.uint8)
            print("Can not find 'model', false image")
         real_img = self.transform(inputs)
         real_img = torch.permute(real_img, dims=(1, 2, 0)).numpy().astype(np.uint8)
         images.append(real_img)
         images.append(gray)
      return images

   def _show(self, images):
      for idx, image in enumerate(images, 0):
         plt.subplot(self.figsize[0], self.figsize[1], idx+1)
         if (image is not None):
            plt.imshow(image, cmap="gray")
         plt.axis("off")
         if (idx < 10):
            plt.title(str(idx))

   def predict(self):
      images = self._predict()
      self._show(images)

   def save(self, filename):
      plt.savefig(filename)
      plt.show()
      plt.close("all")

   def on_epoch_end(self, epoch, logs=None):
      filename = "{}/ep{:>3}.png".format(self.path, epoch)
      self.__init_fig(epoch)
      self.predict()
      self.save(filename)
