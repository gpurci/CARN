#!/usr/bin/python

import torch
from torchvision.transforms import v2
import torch.nn.functional as F
import torchvision.transforms.v2.functional as vF

class TTAImageGenerator():
   def __init__(self, model, select_mixt, **conf):
      self.model = model
      self.__select_mixt(select_mixt)
      self.conf = conf

   @staticmethod
   def help():
      info_help = """TTAImageGenerator methods: basic, mixt,
   mirror, translate, rotate, 
   gray, adjust_brightness, adjust_contrast, adjust_saturation, adjust_hue, 
   """
      print(info_help)

   def __select_mixt(self, select_mixt):
      self.fn_mixt = []

      if ("basic" in select_mixt):
         self.fn_mixt.append(self.basic)
      if ("mirror" in select_mixt):
         self.fn_mixt.append(self.mirror)
      if ("translate" in select_mixt):
         self.fn_mixt.append(self.translate)
      if ("gray" in select_mixt):
         self.fn_mixt.append(self.gray)
      if ("rotate" in select_mixt):
         self.fn_mixt.append(self.rotate)
      if ("adjust_brightness" in select_mixt):
         self.fn_mixt.append(self.adjust_brightness)
      if ("adjust_contrast" in select_mixt):
         self.fn_mixt.append(self.adjust_contrast)
      if ("adjust_saturation" in select_mixt):
         self.fn_mixt.append(self.adjust_saturation)
      if ("adjust_hue" in select_mixt):
         self.fn_mixt.append(self.adjust_hue)
      if ("mixt" in select_mixt):
         self.fn_mixt.append(self.mixt)

   def select(self, tta_type: str):
      if   (tta_type == "mirror"):
         self.__tta_fn = self.mirror
      elif (tta_type == "translate"):
         self.__tta_fn = self.translate
      elif (tta_type == "gray"):
         self.__tta_fn = self.gray
      elif (tta_type == "rotate"):
         self.__tta_fn = self.rotate
      elif (tta_type == "adjust_brightness"):
         self.__tta_fn = self.adjust_brightness
      elif (tta_type == "adjust_contrast"):
         self.__tta_fn = self.adjust_contrast
      elif (tta_type == "adjust_saturation"):
         self.__tta_fn = self.adjust_saturation
      elif (tta_type == "adjust_hue"):
         self.__tta_fn = self.adjust_hue
      elif (tta_type == "mixt"):
         self.__tta_fn = self.mixt
      else:
         self.__tta_fn = self.basic

   def basic(self, inputs):
      return self.model(inputs).detach().cpu()

   def mirror(self, inputs):
      base_pred  = self.model(inputs).detach().cpu()
      # 
      hflip_pred = self.model(vF.hflip(inputs)).detach().cpu()
      hflip_pred = vF.hflip(hflip_pred)
      # 
      vflip_pred = self.model(vF.vflip(inputs)).detach().cpu()
      vflip_pred = vF.vflip(vflip_pred)
      # 
      predicted  = (base_pred + hflip_pred + vflip_pred) / 3
      return predicted

   def translate(self, inputs): # TO DO: read from config
      # base prediction
      base_pred  = self.model(inputs).detach().cpu()
      # translate
      pad = 2
      self.model.setSize((3, 36, 36))
      padded_inputs = F.pad(inputs, (pad,)*4, "reflect")
      pad_pred      = self.model(padded_inputs).detach().cpu()
      pad_pred      = pad_pred[:, :, 2:-2, 2:-2]
      self.model.setSize((3, 32, 32))
      return (base_pred + pad_pred) / 2

   def gray(self, inputs):
      predicted  = self.model(inputs).detach().cpu()
      predicted += self.model(vF.to_grayscale(inputs, num_output_channels=3)).detach().cpu()
      return predicted / 2

   def rotate(self, inputs):
      basse_pred  = self.model(inputs).detach().cpu()
      rot45_pred  = self.model(vF.rotate(inputs, angle=45)).detach().cpu()
      rot45_pred  = vF.rotate(rot45_pred, angle=-45)
      rot_45_pred = self.model(vF.rotate(inputs, angle=-45)).detach().cpu()
      rot_45_pred = vF.rotate(rot_45_pred, angle=45)
      return (basse_pred + rot45_pred + rot_45_pred) / 3

   def adjust_brightness(self, inputs):
      predicted  = self.model(inputs).detach().cpu()
      predicted += self.model(vF.adjust_brightness(inputs, brightness_factor=0.7)).detach().cpu()
      predicted += self.model(vF.adjust_brightness(inputs, brightness_factor=1.3)).detach().cpu()
      return predicted / 3

   def adjust_contrast(self, inputs):
      predicted  = self.model(inputs).detach().cpu()
      predicted += self.model(vF.adjust_contrast(inputs, contrast_factor=0.7)).detach().cpu()
      predicted += self.model(vF.adjust_contrast(inputs, contrast_factor=1.3)).detach().cpu()
      return predicted / 3

   def adjust_saturation(self, inputs):
      predicted  = self.model(inputs).detach().cpu()
      predicted += self.model(vF.adjust_saturation(inputs, saturation_factor=0.7)).detach().cpu()
      predicted += self.model(vF.adjust_saturation(inputs, saturation_factor=1.3)).detach().cpu()
      return predicted / 3

   def adjust_hue(self, inputs):
      predicted  = self.model(inputs).detach().cpu()
      predicted += self.model(vF.adjust_hue(inputs, hue_factor= 0.15)).detach().cpu()
      predicted += self.model(vF.adjust_hue(inputs, hue_factor=-0.15)).detach().cpu()
      return predicted / 3

   def mixt(self, inputs):
      logits_list = []
      for fn in self.fn_mixt:
         logits_list.append(fn(inputs))
      return torch.stack(logits_list).mean(0)

   def __get_model_device(self):
      try:
         parameter = next(self.model.parameters())
         device = parameter.device
      except:
         device = None
      return device

   def __call__(self, x):
      return self.__tta_fn(x)
