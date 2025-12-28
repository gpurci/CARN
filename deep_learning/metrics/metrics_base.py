#!/usr/bin/python

class MetricsBase(object):
   def __init__(self, name=""):
      """"""
      self.name = name

   def __call__(self, y_pred, y):
      raise NameError("The '__call__' is not implemented '{}'".format(self.name))
