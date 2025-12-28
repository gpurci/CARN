#!/usr/bin/python

from metrics.metrics_base import *

class MetricsList(object):
   def __init__(self, metrics_fn:dict):
      """"""
      if (metrics_fn is not None):
         if (isinstance(metrics_fn, (dict))):
            for key in metrics_fn.keys():
               if (not isinstance(metrics_fn[key], type(MetricsBase()))):
                  raise NameError("The metric: '{}' is type '{}' not as 'Metrics' object".format(key, type(metrics_fn[key])))
         else:
            raise NameError("The argument need to be as 'Dict' object, but is '{}'".format(type(metrics_fn)))
      else:
         metrics_fn = {} # default is empty dict
      #
      self.__metrics_fn = metrics_fn
      self.__logs = {}
      self.count  = 0

   def __update_logs(self, key, metric):
      if (self.count == 0):
         self.__logs.update({key: metric})
      else:
         self.__logs[key] += metric

   def __unpack_kwargs(self, prefix, **kw):
      logs = {}
      # update with kwarg logs
      for key in kw.keys():
         metric = kw[key]
         key    = prefix+key
         logs[key] = metric
         self.__update_logs(key, metric)
      return logs

   def __unpack_metrics(self, logs, prefix, y, y_pred):
      for key in self.__metrics_fn.keys():
         metric = self.__metrics_fn[key](y, y_pred)
         key    = prefix+key
         logs[key] = metric
         self.__update_logs(key, metric)
      self.count += 1
      return logs

   def __call__(self, y, y_pred, prefix="", **kw):
      logs = self.__unpack_kwargs(prefix, **kw)
      # perform metrics from metric list
      logs = self.__unpack_metrics(logs, prefix, y, y_pred)
      return logs

   def logs(self):
      logs = {}
      for key in self.__logs.keys():
         self.__logs[key] /= self.count
      logs = self.__logs
      self.__logs = {}
      self.count  = 0
      return logs
