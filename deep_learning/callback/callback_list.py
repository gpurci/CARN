#!/usr/bin/python

from livelossplot import PlotLossesKeras
from callback.callback_base import *

class CallbacksList(object):
   def __init__(self, callbacks):
      """"""
      if (callbacks is not None):
         if (isinstance(callbacks, list)):
               for i, callback in enumerate(callbacks, 0):
                  if (not isinstance(callback, (type(CallbacksBase()), type(PlotLossesKeras())))):
                     raise NameError("The callback: '{}' is type '{}' not as 'Callbacks' object".format(i, type(callback)))
         else:
            raise NameError("The argument need to be as 'List' object, but is '{}'".format(type(callbacks)))
      else:
         callbacks = []
      #
      self.__callbacks = callbacks

   def on_batch_begin(self, batch, logs=None):
      for callback in self.__callbacks:
         callback.on_batch_begin(batch, logs)

   def on_batch_end(self, batch, logs=None):
      for callback in self.__callbacks:
         callback.on_batch_end(batch, logs)

   def on_epoch_begin(self, epoch, logs=None):
      for callback in self.__callbacks:
         callback.on_epoch_begin(epoch, logs)

   def on_epoch_end(self, epoch, logs=None):
      for callback in self.__callbacks:
         callback.on_epoch_end(epoch, logs)

   def on_predict_batch_begin(self, batch, logs=None):
      for callback in self.__callbacks:
         callback.on_predict_batch_begin(batch, logs)

   def on_predict_batch_end(self, batch, logs=None):
      for callback in self.__callbacks:
         callback.on_predict_batch_end(batch, logs)

   def on_predict_begin(self, logs=None):
      for callback in self.__callbacks:
         callback.on_predict_begin(logs)

   def on_predict_end(self, logs=None):
      for callback in self.__callbacks:
         callback.on_predict_end(logs)

   def on_test_batch_begin(self, batch, logs=None):
      for callback in self.__callbacks:
         callback.on_test_batch_begin(batch, logs)

   def on_test_batch_end(self, batch, logs=None):
      for callback in self.__callbacks:
         callback.on_test_batch_end(batch, logs)

   def on_test_begin(self, logs=None):
      for callback in self.__callbacks:
         callback.on_test_begin(logs)

   def on_test_end(self, logs=None):
      for callback in self.__callbacks:
         callback.on_test_end(logs)

   def on_train_batch_begin(self, batch, logs=None):
      for callback in self.__callbacks:
         callback.on_train_batch_begin(batch, logs)

   def on_train_batch_end(self, batch, logs=None):
      for callback in self.__callbacks:
         callback.on_train_batch_end(batch, logs)

   def on_train_begin(self, logs=None):
      for callback in self.__callbacks:
         callback.on_train_begin(logs)

   def on_train_end(self, logs=None):
      for callback in self.__callbacks:
         callback.on_train_end(logs)
