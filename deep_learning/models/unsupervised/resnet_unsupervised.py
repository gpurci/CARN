#!/usr/bin/python

import torch
from torch import nn
from collections import OrderedDict

from sys_function import * # este in root
sys_remove_modules("layers.resnet_blocks.identity_resnet_module")
sys_remove_modules("layers.resnet_blocks.input_by_stride")
sys_remove_modules("layers.identity_conv2d_downsample")
sys_remove_modules("layers.conv2d_downsample")

from layers.resnet_blocks.identity_resnet_module import *
from layers.resnet_blocks.input_by_stride import *
from layers.identity_conv2d_downsample import *
from layers.conv2d_downsample import *
from layers.conv2d_upsample import *

class ResNetUnsupervised(nn.Module):
    def __init__(self, name, initializer=None, **conf):
        super().__init__()
        self.name  = name
        self.input   = self._unpack_input(**conf)
        self.encode  = self._unpack_encode(**conf)
        self.decode  = self._unpack_decode(**conf)
        in_features, out_features = conf.get("Output")
        self.fc      = nn.Conv2d(
                                    in_features,
                                    out_features,
                                    kernel_size=1,
                                    stride=1,
                                    padding="same",
                                    bias=False,
                                    groups=1,
                                )
        self.tanh    = nn.Tanh()

    def _unpack_input(self, **conf):
        tmp_conf = conf.get("Input")
        img_channels = tmp_conf.get("img_channels")
        out_channels = tmp_conf.get("out_channels")
        kernel_size = tmp_conf.get("kernel_size")
        stride = tmp_conf.get("stride")

        layer = InputStride(img_channels, out_channels, kernel_size, stride)
        return layer

    def _unpack_encode(self, **conf):
        encode_conf = conf.get("encode")
        layers = []
        for key in encode_conf.keys():
            layer_conf   = encode_conf.get(key)
            block_layers = self._unpack_encode_blocks(key, **layer_conf)
            layers.extend(block_layers)
        return nn.Sequential(OrderedDict(layers))

    def _unpack_encode_blocks(self, key, **conf):
        in_channels = conf.get("in_channels", None)
        expansion   = conf.get("expansion", 4)
        stride      = conf.get("stride", 1)
        intermediate_channels = conf.get("intermediate_channels", None)
        num_residual_blocks   = conf.get("num_residual_blocks", 1)
        out_channels = intermediate_channels * expansion
        #print("name {}, in_channels {}, out_channels {}, expansion {}, stride {}, intermediate_channels {}, num_residual_blocks {}".format(key, in_channels, out_channels, expansion, stride, intermediate_channels, num_residual_blocks))
        layers = []
        for idx in range(num_residual_blocks):
            identity_downsample = None
            if ((stride != 1) or (in_channels != out_channels)):
                identity_downsample = IdentityConv2dDownSample(in_channels, out_channels, stride=stride)
            layer = IdentityResNetModule(in_channels, intermediate_channels, 
                                            expansion=expansion, identity_downsample=identity_downsample, stride=stride)
            in_channels = out_channels
            stride = 1
            block_name = "{}_{}".format(key, idx)
            layers.append((block_name, layer))
        return layers

    def _unpack_decode(self, **conf):
        decode_conf = conf.get("decode")
        layers = []
        for key in decode_conf.keys():
            layer_conf   = decode_conf.get(key)
            block_layers = self._unpack_decode_blocks(key, **layer_conf)
            layers.extend(block_layers)
        return nn.Sequential(OrderedDict(layers))

    def _unpack_decode_blocks(self, key, **conf):
        in_channels = conf.get("in_channels", None)
        expansion   = conf.get("expansion", 4)
        stride      = conf.get("stride", 1)
        intermediate_channels = conf.get("intermediate_channels", None)
        num_residual_blocks   = conf.get("num_residual_blocks", 1)
        out_channels = intermediate_channels * expansion
        #print("name {}, in_channels {}, out_channels {}, expansion {}, stride {}, intermediate_channels {}, num_residual_blocks {}".format(key, in_channels, out_channels, expansion, stride, intermediate_channels, num_residual_blocks))
        layers = []
        for idx in range(num_residual_blocks):
            identity_downsample = None
            if (stride != 1):
                layer  = Conv2dUpSample(in_channels, kernel_size=3, stride=stride)
                stride = 1
                block_name = "{}_{}_upsample".format(key, idx)
                layers.append((block_name, layer))
            if (in_channels != out_channels):
                identity_downsample = IdentityConv2dDownSample(in_channels, out_channels, stride=stride)
            layer = IdentityResNetModule(in_channels, intermediate_channels, 
                                            expansion=expansion, identity_downsample=identity_downsample, stride=1)
            in_channels = out_channels
            block_name = "{}_{}".format(key, idx)
            layers.append((block_name, layer))
        return layers

    def reset_parameters(self):
        self.input.reset_parameters()
        self.fc.reset_parameters()
        for layer in self.encode:
            layer.reset_parameters()
        for layer in self.decode:
            layer.reset_parameters()

    def forward(self, x):
        x = self.input(x)
        x = self.encode(x)
        x = self.decode(x)
        x = self.fc(x)
        x = self.tanh(x)
        return x

"""
Input=(img_channels, out_channels, kernel_size, stride)
encode={body_name={in_channels, expansion, stride, intermediate_channels, num_residual_blocks}}
Output=(in_features, out_features)
"""
