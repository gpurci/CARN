#!/usr/bin/python

import torch
from torch import nn
import numpy as np

class AttUnetInput(nn.Module):
   def __init__(self, img_shape, out_channels, frame_dim, num_heads):
      super().__init__()
      img_channels = img_shape[0]
      self.conv1 = nn.Conv2d(
            img_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            groups=1,
         )
      self.bn1  = nn.BatchNorm2d(out_channels)
      # 
      img_size  = np.prod(img_shape)
      embed_dim = np.prod(frame_dim)*img_channels
      #
      #                    B, H/frame0,                   frame0,       W/frame1,                   frame1,       Ch
      self.hin0_shape  = (-1, img_shape[1]//frame_dim[0], frame_dim[0], img_shape[2]//frame_dim[1], frame_dim[1], img_shape[0])
      #                    B, Seq=H'*W',                                             d_model=frame0*frame1*Ch
      self.hin1_shape  = (-1, img_shape[1]//frame_dim[0]*img_shape[2]//frame_dim[1], frame_dim[0]*frame_dim[1]*img_shape[0])

      #
      #                 H/frame0,                        W/frame1,                   frame0,       frame1,       Ch
      self.hout_shape = (-1, img_shape[1]//frame_dim[0], img_shape[2]//frame_dim[1], frame_dim[0], frame_dim[1], img_shape[0])
      # 
      self.img_shape  = (-1, *img_shape)

      self.transformer_encode0 = nn.TransformerEncoderLayer(embed_dim, num_heads, 
         dim_feedforward=int(embed_dim*2), dropout=0.2, activation=nn.LeakyReLU(0.2), 
         layer_norm_eps=1e-05, batch_first=True, 
         norm_first=False, bias=False, device=None, dtype=None)
      self.transformer_encode1 = nn.TransformerEncoderLayer(embed_dim, num_heads, 
         dim_feedforward=int(embed_dim*2), dropout=0.2, activation=nn.LeakyReLU(0.2), 
         layer_norm_eps=1e-05, batch_first=True, 
         norm_first=False, bias=False, device=None, dtype=None)
      self.transformer_encode2 = nn.TransformerEncoderLayer(embed_dim, num_heads, 
         dim_feedforward=int(embed_dim*2), dropout=0.15, activation=nn.LeakyReLU(0.2), 
         layer_norm_eps=1e-05, batch_first=True, 
         norm_first=False, bias=False, device=None, dtype=None)
      self.transformer_encode3 = nn.TransformerEncoderLayer(embed_dim, num_heads, 
         dim_feedforward=int(embed_dim*2), dropout=0.15, activation=nn.LeakyReLU(0.2), 
         layer_norm_eps=1e-05, batch_first=True, 
         norm_first=False, bias=False, device=None, dtype=None)
      self.transformer_encode4 = nn.TransformerEncoderLayer(embed_dim, num_heads, 
         dim_feedforward=int(embed_dim*2), dropout=0.1, activation=nn.LeakyReLU(0.2), 
         layer_norm_eps=1e-05, batch_first=True, 
         norm_first=False, bias=False, device=None, dtype=None)
      self.transformer_encode5 = nn.TransformerEncoderLayer(embed_dim, num_heads, 
         dim_feedforward=int(embed_dim*2), dropout=0.1, activation=nn.LeakyReLU(0.2), 
         layer_norm_eps=1e-05, batch_first=True, 
         norm_first=False, bias=False, device=None, dtype=None)
      self.activ_fn = nn.LeakyReLU(0.2)

   def reset_parameters(self):
      self.bn1.reset_parameters()
      self.conv1.reset_parameters()
      self.transformer_encode.reset_parameters()

   def permute_in(self, x):
      # input_shape -> B, Ch, H, W
      # permute0    -> B, H, W, Ch
      # reshape1    -> B, H/frame0, frame0, W/frame1, frame1, Ch
      # reshape1    -> B, H', frame0, W', frame1, Ch
      # permute2    -> B, H', W', frame0, frame1, Ch
      # reshape3    -> B, Seq=H'*W', d_model=frame0*frame1*Ch
      # output 3    -> B, Seq, d_model
      #
      # permute  -> B, H, W, Ch
      x = torch.permute(x, dims=(0, 2, 3, 1))
      # reshape  -> B, H', frame0, W', frame1, Ch
      x = x.reshape(*self.hin0_shape)
      # permute  -> B, H', W', frame0, frame1, Ch
      x = torch.permute(x, dims=(0, 1, 3, 2, 4, 5))
      # reshape  -> B, Seq=H'*W', d_model=frame0*frame1*Ch
      # reshape  -> B, Seq, d_model
      x = x.reshape(*self.hin1_shape)

      return x

   def permute_out(self, x):
      # input_shape0 -> B, Seq, d_model
      # input_shape0 -> B, H'* W', frame0 *frame1 *Ch 
      # reshape1     -> B, H', W', frame0, frame1, Ch
      # permute2     -> B, Ch, H', frame0, W', frame1
      # reshape3     -> B, Ch, H'* frame0, W'* frame1
      # output 3     -> B, Ch, H, W
      #
      # reshape     -> B, H', W', frame0, frame1, Ch
      x = x.reshape(*self.hout_shape)
      # permute     -> B, Ch, H', frame0, W', frame1
      x = torch.permute(x, dims=(0, 5, 1, 3, 2, 4))
      # reshape     -> B, Ch, H'* frame0, W'* frame1
      # output      -> B, Ch, H, W
      x = x.reshape(*self.img_shape)
      return x

   def forward(self, x):
      x = self.permute_in(x)
      x = self.transformer_encode0(x)
      x = self.transformer_encode1(x)
      x = self.transformer_encode2(x)
      #x = self.transformer_encode3(x)
      #x = self.transformer_encode4(x)
      #x = self.transformer_encode5(x)
      x = self.permute_out(x)

      x = self.conv1(x)
      x = self.bn1(x)
      x = self.activ_fn(x)
      return x
