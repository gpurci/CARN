#!/usr/bin/python

class MetricsList(object):
   def __init__(self, metrics_fn):
      """"""
      if (metrics_fn is not None):
         if (isinstance(metrics_fn, (dict))):
            for key in metrics_fn.keys():
               if (not isinstance(metrics_fn[key], Metrics)):
                  raise NameError("The metric: '{}' is type '{}' not as 'Metrics' object".format(key, type(metrics_fn[key])))
         else:
            raise NameError("The argument need to be as 'Dict' object, but is '{}'".format(type(metrics_fn)))
      else:
         pass
      #
      self.__metrics_fn = metrics_fn

   def __call__(self, y, y_pred):
      logs = {}
      if (self.__metrics_fn is not None):
         for key in self.__metrics_fn.keys():
            metric = self.__metrics_fn[key](y, y_pred)
            logs[key] = metric
      return logs
