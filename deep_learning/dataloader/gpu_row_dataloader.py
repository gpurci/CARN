#!/usr/bin/python

import os

import torch
from torch.utils.data import Dataset

import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

class GpuRowDataLoader():
   def __init__(self, path, inputs, targets, train, mean, std, batch_size=500, aug=None):
      data_path = os.path.join(path, "train.pt" if train else "test.pt")
      if (not os.path.exists(data_path)):
         images = torch.tensor(inputs)
         labels = torch.tensor(targets)
         torch.save({"images": images, "labels": labels}, data_path)

      data = torch.load(data_path, map_location=torch.device("cuda"))
      self.images, self.labels = data["images"], data["labels"]
      # It's faster to load+process uint8 data than to load preprocessed fp16 data
      self.images = self.images.half().permute(0, 3, 1, 2).to(memory_format=torch.channels_last)

      self.normalize = T.Normalize(mean, std)
      self.proc_images = {} # Saved results of image processing to be done on the first epoch
      self.epoch = 0

      self.aug = aug or {}
      for k in self.aug.keys():
         assert k in ["flip", "translate"], "Unrecognized key: %s" % k

      self.batch_size = batch_size
      self.drop_last = train
      self.shuffle = train

   def __len__(self):
      return len(self.images)//self.batch_size if self.drop_last else ceil(len(self.images)/self.batch_size)

   def __iter__(self):

      if (self.epoch == 0):
         images = self.proc_images["norm"] = self.normalize(self.images)
         # Pre-flip images in order to do every-other epoch flipping scheme
         if (self.aug.get("flip", False)):
            images = self.proc_images["flip"] = batch_flip_lr(images)
         # Pre-pad images to save time when doing random translation
         pad = self.aug.get("translate", 0)
         if (pad > 0):
            self.proc_images["pad"] = F.pad(images, (pad,)*4, "reflect")

      if   (self.aug.get("translate", 0) > 0):
         images = batch_crop(self.proc_images["pad"], self.images.shape[-2])
      elif (self.aug.get("flip", False)):
         images = self.proc_images["flip"]
      else:
         images = self.proc_images["norm"]
      # Flip all images together every other epoch. This increases diversity relative to random flipping
      if (self.aug.get("flip", False)):
         if (self.epoch % 2 == 1):
            images = images.flip(-1)

      self.epoch += 1

      indices = (torch.randperm if self.shuffle else torch.arange)(len(images), device=images.device)
      for i in range(len(self)):
         idxs = indices[i*self.batch_size:(i+1)*self.batch_size]
         yield (images[idxs], self.labels[idxs])

def batch_flip_lr(inputs):
   flip_mask = (torch.rand(len(inputs), device=inputs.device) < 0.5).view(-1, 1, 1, 1)
   return torch.where(flip_mask, inputs.flip(-1), inputs)

def batch_crop(images, crop_size):
   r = (images.size(-1) - crop_size)//2
   shifts = torch.randint(-r, r+1, size=(len(images), 2), device=images.device)
   images_out = torch.empty((len(images), 3, crop_size, crop_size), device=images.device, dtype=images.dtype)
   # The two cropping methods in this if-else produce equivalent results, but the second is faster for r > 2.
   if (r <= 2):
      for sy in range(-r, r+1):
         for sx in range(-r, r+1):
            mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
            images_out[mask] = images[mask, :, r+sy:r+sy+crop_size, r+sx:r+sx+crop_size]
   else:
      images_tmp = torch.empty((len(images), 3, crop_size, crop_size+2*r), device=images.device, dtype=images.dtype)
      for s in range(-r, r+1):
         mask = (shifts[:, 0] == s)
         images_tmp[mask] = images[mask, :, r+s:r+s+crop_size, :]
      for s in range(-r, r+1):
         mask = (shifts[:, 1] == s)
         images_out[mask] = images_tmp[mask, :, :, r+s:r+s+crop_size]
   return images_out

