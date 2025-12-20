#!/usr/bin/python

import torch
import torch.nn.init as init

def xavier(parameters):
   for l_params in parameters:
      if (l_params is not None):
         if (l_params.dim() > 1):
            init.xavier_uniform_(l_params)
         else:
            init.zeros_(l_params)
