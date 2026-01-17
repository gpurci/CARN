#!/usr/bin/python

import torch
from torch import nn, Tensor

# source: https://github.com/KellerJordan/cifar10-airbench/blob/master/airbench94_muon.py

class ApplyWhiten2d(nn.Module):
   def __init__(self, model, in_chanels):
      super().__init__()
      self.model = model
      self.head  = self.model.fc
      # TO DO: add documentation about whitening operation
      # if change whiten_kernel_size, change the pad
      whiten_kernel_size = 2 # This implies a 2×2 spatial patch is used for the whitening operation
      # 3 → number of input channels (RGB) (in_chanels)
      # whiten_kernel_size**2 → spatial positions per channel (4)
      # 2 → two components per value (commonly real + imaginary, or sine + cosine, 
      #     or positive/negative projections depending on the implementation)
      whiten_width = 2 * int(in_chanels) * whiten_kernel_size**2 
      self.whiten    = nn.Conv2d(3, whiten_width, whiten_kernel_size, stride=1, padding=0, bias=True)
      self.whiten.weight.requires_grad = False
      #self.reduce_ch = nn.Conv2d(whiten_width, 3, kernel_size=1, stride=1, padding=0, bias=False)
      # (padding_left,padding_right, padding_top,padding_bottom)
      self.pad = (0, 1, 0, 1)

      for mod in self.model.layers:
         if isinstance(mod, nn.BatchNorm2d):
            mod.float()
         else:
            mod.half()
      self.head.half()

   def reset_parameters(self):
      for m in self.modules():
         if (type(m) in (nn.Conv2d, nn.BatchNorm2d)):
            m.reset_parameters()
      w = self.head.weight.data
      w *= 1. / w.std()

   @staticmethod
   def transform(inputs, mean, std):
      inputs = torch.from_numpy(inputs)
      inputs = (inputs - mean) / std
      inputs = torch.permute(inputs, dims=(0, 3, 1, 2))
      return inputs

   @staticmethod
   def get_patches(inputs, patch_shape):
      c, (h, w) = inputs.shape[1], patch_shape
      return inputs.unfold(2,h,1).unfold(3,w,1).transpose(1,3).reshape(-1,c,h,w).float()

   @staticmethod
   def get_whitening_parameters(patches):
      n,c,h,w = patches.shape
      patches_flat = patches.view(n, -1)
      est_patch_covariance = (patches_flat.T @ patches_flat) / n
      eigenvalues, eigenvectors = torch.linalg.eigh(est_patch_covariance, UPLO="U")
      return eigenvalues.flip(0).view(-1, 1, 1, 1), eigenvectors.T.reshape(c*h*w,c,h,w).flip(0)

   def init_whiten(self, inputs, eps=5e-4):
      patches = ApplyWhiten2d.get_patches(inputs, self.whiten.weight.shape[2:])
      eigenvalues, eigenvectors = ApplyWhiten2d.get_whitening_parameters(patches)
      eigenvectors_scaled = eigenvectors / torch.sqrt(eigenvalues + eps)
      self.whiten.weight.data[:] = torch.cat((eigenvectors_scaled, -eigenvectors_scaled))

   def forward(self, x, whiten_bias_grad=False):
      b = self.whiten.bias
      b_used = b if whiten_bias_grad else b.detach()
      b_used = b_used.to(dtype=x.dtype)
      x = nn.functional.pad(x, self.pad, "constant", 0)
      x = nn.functional.conv2d(x, self.whiten.weight.to(dtype=x.dtype), b_used)
      #x = self.reduce_ch(x)
      x = self.model.layers(x)
      x = x.view(len(x), -1)
      return self.head(x) / x.size(-1)
