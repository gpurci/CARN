#!/usr/bin/python

import torch

class Muon(torch.optim.Optimizer):
   def __init__(self, params, lr=1e-3, momentum=0, nesterov=False, 
            ns_coefficients=(3.4445, -4.775, 2.0315), ns_steps=3, eps=1e-7):
      """
source: https://github.com/KellerJordan/cifar10-airbench/blob/master/airbench94_muon.py
      """
      if (lr < 0.0):
         raise ValueError(f"Invalid learning rate: {lr}")
      if (momentum < 0.0):
         raise ValueError(f"Invalid momentum value: {momentum}")
      if (nesterov and (momentum <= 0)):
         raise ValueError("Nesterov momentum requires a momentum")
      defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
      super().__init__(params, defaults)
      self.ns_coefficients = ns_coefficients
      self.ns_steps = ns_steps
      self.eps = eps

   def step(self):
      for group in self.param_groups:
         lr = group["lr"]
         momentum = group["momentum"]
         for p in group["params"]:
            g = p.grad
            if g is None:
               continue
            state = self.state[p]

            if ("momentum_buffer" not in state.keys()):
               state["momentum_buffer"] = torch.zeros_like(g)
            buf = state["momentum_buffer"]
            buf.mul_(momentum).add_(g)
            g = g.add(buf, alpha=momentum) if group["nesterov"] else buf

            p.data.mul_(len(p.data)**0.5 / p.data.norm()) # normalize the weight
            update = zeropower_via_newtonschulz5(g.reshape(len(g), -1), self.ns_coefficients, self.ns_steps, self.eps)
            update = update.view(g.shape) # whiten the update
            p.data.add_(update, alpha=-lr) # take a step

@torch.compile
def zeropower_via_newtonschulz5(G, ns_coefficients, ns_steps=3, eps=1e-7):
   """
   Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
   quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
   of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
   zero even beyond the point where the iteration no longer converges all the way to one everywhere
   on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
   where S' is diagonal with S_{ii}' \sim Uniform(0.5, 1.5), which turns out not to hurt model
   performance at all relative to UV^T, where USV^T = G is the SVD.
   """
   error_msg = "Error: dim of 'grad' is '{}' should be '2'".format(len(G.shape))
   assert (len(G.shape) == 2), error_msg
   a, b, c = ns_coefficients
   X = G.bfloat16()
   X /= (X.norm() + eps) # ensure top singular value <= 1
   if (G.size(0) > G.size(1)):
      X = X.T
   for _ in range(ns_steps):
      A = X @ X.T
      B = b * A + c * A @ A
      X = a * X + B @ X
   if (G.size(0) > G.size(1)):
      X = X.T
   return X
