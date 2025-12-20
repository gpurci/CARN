#!/usr/bin/python

import torch
from torch import nn

class Conv2dUpSample(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=2):
        super().__init__()
        assert (stride > 1), "stride should be more that 1"
        assert (isinstance(stride, int)), "stride should be int"
        self.stride = stride
        # convolution layer
        self.conv1 = nn.Conv2d(
                in_channels,
                in_channels*(stride**2),
                kernel_size=kernel_size,
                stride=1,
                padding="same",
                bias=False,
                groups=in_channels,
            )
        self.bn1 = nn.BatchNorm2d(in_channels)
        # shape -> B0, Ch1, stride2, stride3, H4, W5
        # perm  -> B0, Ch1, stride2, H4, stride3, W5
        self.perm = (0, 1, 4, 2, 5, 3)

    def reset_parameters(self):
        self.bn1.reset_parameters()
        self.conv1.reset_parameters()

    def permute(self, x_inputs):
        # input_shape -> B, Ch * stride^2,      H, W
        # reshape     -> B, Ch, stride, stride, H, W
        input_shape = x_inputs.size()
        x_inputs = x_inputs.reshape(input_shape[0], -1, self.stride, self.stride, input_shape[2], input_shape[3])
        # permute  -> B, Ch, stride, H, stride, W
        x_inputs = torch.permute(x_inputs, dims=self.perm)
        # reshape  -> B, Ch, stride*H, stride*W
        x_inputs = x_inputs.reshape(input_shape[0], -1, self.stride*input_shape[2], self.stride*input_shape[3])
        return x_inputs
    
    def forward(self, x):
        x = self.conv1(x) # shape -> B, Ch * stride^2, H, W
        x = self.permute(x)
        x = self.bn1(x)
        return x
