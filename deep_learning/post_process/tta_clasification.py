#!/usr/bin/python

import torch
from torchvision.transforms import v2
import torch.nn.functional as F
import torchvision.transforms.v2.functional as vF

class TTAClasification():
   def __init__(self, model, select_mixt, **conf):
      self.model = model
      self.__select_mixt(select_mixt)
      self.conf = conf

   @staticmethod
   def help():
      info_help = """TTA methods: basic, mixt,
   mirror, translate, mirroring_and_translate, rotate, 
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
      if ("mirroring_and_translate" in select_mixt):
         self.fn_mixt.append(self.mirror_translate)
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
      elif (tta_type == "mirroring_and_translate"):
         self.__tta_fn = self.mirror_translate
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
      predicted  = self.model(inputs).detach().cpu()
      predicted += self.model(vF.hflip(inputs)).detach().cpu()
      predicted += self.model(vF.vflip(inputs)).detach().cpu()
      return predicted

   def translate(self, inputs): # TO DO: read from config
      pad = 1
      padded_inputs = F.pad(inputs, (pad,)*4, "reflect")
      inputs_pad_list = [
         padded_inputs[:, :, 0:32, 0:32],
         padded_inputs[:, :, 2:34, 2:34],
      ]
      logits_list = [self.basic(inputs_translate) for inputs_translate in inputs_pad_list]
      logits = torch.stack(logits_list).mean(0)
      return logits

   def mirror_translate(self, inputs):
      logits = self.mirror(inputs)
      pad = 1
      padded_inputs = F.pad(inputs, (pad,)*4, "reflect")
      inputs_pad_list = [
         padded_inputs[:, :, 0:32, 0:32],
         padded_inputs[:, :, 2:34, 2:34],
      ]
      logits_translate_list = [self.mirror(inputs_translate) for inputs_translate in inputs_pad_list]
      logits_translate = torch.stack(logits_translate_list).mean(0)
      return logits + logits_translate

   def gray(self, inputs):
      predicted  = self.model(inputs).detach().cpu()
      predicted += self.model(vF.to_grayscale(inputs, num_output_channels=3)).detach().cpu()
      return predicted

   def rotate(self, inputs):
      predicted  = self.model(inputs).detach().cpu()
      predicted += self.model(vF.rotate(inputs, angle=45)).detach().cpu()
      predicted += self.model(vF.rotate(inputs, angle=-45)).detach().cpu()
      return predicted

   def adjust_brightness(self, inputs):
      predicted  = self.model(inputs).detach().cpu()
      predicted += self.model(vF.adjust_brightness(inputs, brightness_factor=0.7)).detach().cpu()
      predicted += self.model(vF.adjust_brightness(inputs, brightness_factor=1.3)).detach().cpu()
      return predicted

   def adjust_contrast(self, inputs):
      predicted  = self.model(inputs).detach().cpu()
      predicted += self.model(vF.adjust_contrast(inputs, contrast_factor=0.7)).detach().cpu()
      predicted += self.model(vF.adjust_contrast(inputs, contrast_factor=1.3)).detach().cpu()
      return predicted

   def adjust_saturation(self, inputs):
      predicted  = self.model(inputs).detach().cpu()
      predicted += self.model(vF.adjust_saturation(inputs, saturation_factor=0.7)).detach().cpu()
      predicted += self.model(vF.adjust_saturation(inputs, saturation_factor=1.3)).detach().cpu()
      return predicted

   def adjust_hue(self, inputs):
      predicted  = self.model(inputs).detach().cpu()
      predicted += self.model(vF.adjust_hue(inputs, hue_factor= 0.15)).detach().cpu()
      predicted += self.model(vF.adjust_hue(inputs, hue_factor=-0.15)).detach().cpu()
      return predicted

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
