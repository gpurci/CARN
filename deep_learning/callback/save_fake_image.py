#!/usr/bin/python

import torch
from torch import nn

from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path

from callback.callback_base import *

class SaveFakeImage(CallbacksBase):
   def __init__(self, model, dataset, transform, device, size=1, path="", freq=1):
      self.setModel(model)
      self.dataset   = dataset
      self.transform = transform
      self.device  = device
      self.figsize = (size, 2) # (rows, column)
      self.path    = path
      self.freq    = freq
      
      Path(self.path).mkdir(mode=0o777, parents=True, exist_ok=True)
      filename = "{}/ep_{}.png".format(self.path, "init")
      self.__init_fig("init")
      self.predict_and_put()
      self.save("init", filename)
      self.show()

   def __init_fig(self, epoch):
      # save a image using extension
      fig = plt.figure(figsize=self.figsize)
      fig.set_figheight(5)
      fig.set_figwidth(5)
      # set the spacing between subplots
      fig.suptitle("This is epoch {}".format(epoch), fontsize=16)

      plt.subplots_adjust(left=0.1,
                          bottom=0.1,
                          right=0.9,
                          top=0.9,
                          wspace=0.09,
                          hspace=0.0009)

   def __get_model_device(self):
      try:
         parameter = next(self._model.parameters())
         device = parameter.device
      except:
         device = None
      return device

   def __predict_no_model(self):
      images = []
      for idx in range(self.figsize[0]):
         inputs, _ = self.dataset[idx]
         num_channels = inputs.size(0)
         fake_img = np.zeros((10, 10, num_channels), dtype=np.uint8)
         row_img = self.transform(inputs)
         row_img = torch.permute(row_img, dims=(1, 2, 0)).detach().cpu().numpy().astype(np.uint8)
         images.append(row_img)
         images.append(fake_img)
      return images

   def __predict_model(self):
      images = []
      model_device = self.__get_model_device()
      self._model  = self._model.to(self.device)
      for idx in range(self.figsize[0]):
         inputs, _ = self.dataset[idx]
         inputs = inputs.to(self.device)
         try:
            fake_img = torch.unsqueeze(inputs, dim=0)
            fake_img = self._model(fake_img)
            fake_img = torch.squeeze(fake_img, dim=0)
            fake_img = self.transform(fake_img)
            fake_img = torch.permute(fake_img, dims=(1, 2, 0)).detach().cpu().numpy().astype(np.uint8)
         except Exception as e:
            num_channels = inputs.size(0)
            fake_img = np.zeros((10, 10, num_channels), dtype=np.uint8)
            print("Error prediction, fake image: '{}'".format(e))
         row_img = self.transform(inputs)
         row_img = torch.permute(row_img, dims=(1, 2, 0)).detach().cpu().numpy().astype(np.uint8)
         images.append(row_img)
         images.append(fake_img)
      self._model = self._model.to(model_device)
      return images

   def __put(self, images):
      for idx, image in enumerate(images, 0):
         plt.subplot(self.figsize[0], self.figsize[1], idx+1)
         if (image is not None):
            if   (image.shape[-1] == 1):
               plt.imshow(image, cmap="gray")
            elif (image.shape[-1] == 3):
               plt.imshow(image)
         plt.axis("off")
         if (idx < 2):
            titles = ["Row image", "Generated image"]
            plt.title(titles[idx])

   def predict_and_put(self):
      if (hasattr(self, "_model")):
         images = self.__predict_model()
      else:
         images = self.__predict_no_model()
      self.__put(images)

   def show(self):
      plt.show()
      plt.close("all")

   def save(self, epoch, filename):
      if ((isinstance(epoch, int)) and (epoch % self.freq) == 0):
         plt.savefig(filename)

   def on_epoch_end(self, epoch, logs=None):
      filename = "{}/ep{:>3}.png".format(self.path, epoch)
      self.__init_fig(epoch)
      self.predict_and_put()
      self.save(epoch, filename)
      self.show()
