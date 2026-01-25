#  Callbacks

##  Description

This folder contains utilities for monitoring the training process, persisting intermediate results, and executing external commands invoked during model training.
---

## Project Structure

| File | Description |
|------|--------------|
| `callback_base.py` | This class serves as the base class and defines all fundamental methods that a `callback` is required to implement. |
| `callback_list.py` | This class functions as a container for `callbacks`, accepting a list of `callback` instances as input and invoking them sequentially in first-in, first-out (FIFO) order. |
| `save_model_by_best_key.py` | This class save the model whenever a specified metric in the training logs exceeds its previously recorded value. |
| `save_logs2csv.py` | This class saves the logs to a `csv` file.  |
| `save_fake_image.py` | This class is for image generation, it saves a list of training images concatenated with synthesized images in a folder. |
| `show_logs.py` | This class displays a plot for each log, (training and testing logs are displayed in the same plot). |
| `time_accuracy.py` | This class displays the total training time up to the current epoch and the accuracy of the validation data. |
| `test` | This folder contains the test suite used to validate the functionality of the `callback` components. |

---

## callback_base

This class serves as the base class and defines all essential methods that a `callback` is required to implement. Further details can be found in the `callback_base.py` file.

```python
class CallbacksBase(object):
   def __init__(self,):
      """"""
      pass

   def setModel(self, model):
      self._model = model

   def on_batch_begin(self, batch, logs=None):
      pass

```

---

## callback_list

This class acts as a container for `callbacks`, accepting a list of `callback` instances as input and invoking them in first-in, first-out (FIFO) order. Additional details can be found in `callback_list.py`.

Parameters:
- callbacks: a list of `callback` instances that inherit from `CallbacksBase`.
```python
class CallbacksList(object):
   def __init__(self, callbacks):
      """"""
      if (callbacks is not None):
         if (isinstance(callbacks, list)):
               for i, callback in enumerate(callbacks, 0):
                  if (not isinstance(callback, (type(CallbacksBase())))):
                     raise NameError("The callback: '{}' is type '{}' not as 'Callbacks' object".format(i, type(callback)))
         else:
            raise NameError("The argument need to be as 'List' object, but is '{}'".format(type(callbacks)))
      else:
         callbacks = []
      #
      self.__callbacks = callbacks

   def setModel(self, model):
      for callback in self.__callbacks:
         callback.setModel(model)

   def on_batch_begin(self, batch, logs=None):
      for callback in self.__callbacks:
         callback.on_batch_begin(batch, logs)

```

---

## save_model_by_best_key

This class saves the model whenever a specified metric in the logs exceeds its previously recorded value. Further details are provided in `save_model_by_best_key.py`.

Parameters:
- model (nn.Module): the model to be saved.
- filename (str): the name of the file in which the model will be stored.
- key (str): the metric in the logs used to determine when the model should be saved.
```python
class SaveModelByBestKey(CallbacksBase):
   def __init__(self, model, filename, key="val_acc"):
```

---

## save_logs2csv

This class records training logs to a CSV file. Further details can be found in `save_logs2csv.py`.

Parameters:
- filename (str): the name of the CSV file where the logs will be saved.
```python
class SaveLogs2csv(CallbacksBase):
   def __init__(self, filename):
```

---

## save_fake_image

This class is designed for image generation and saves a set of training images concatenated with synthesized images to a specified folder. Further details can be found in `save_fake_image.py`.

Parameters:
- model (nn.Module): the model used for generating images.
- dataset (torch.utils.data.Dataset): the dataset from which images will be sampled, either normalized or preprocessed according to the model’s requirements.
- transform (torchvision.transforms.v2.Transform): the function used to normalize the data to the 0–255 range.
- device: the device on which the model will be executed.
- size (int): the number of images to extract from the dataset.
- path (str): the folder where the images will be saved.
- freq (int): the interval at which images are saved to path.
```python
class SaveFakeImage(CallbacksBase):
   def __init__(self, model, dataset, transform, device, size=1, path="", freq=1):
```

---

## show_logs

This class visualizes training and testing logs within a single plot. Further details can be found in `show_logs.py`.

Parameters:
- title (str): the title of the plot.
```python
class ShowLogs(CallbacksBase):
   def __init__(self, title):
```

---

## time_accuracy

This class presents the cumulative training time up to the current epoch, along with the accuracy on the validation dataset. Further details are provided in `time_accuracy.py`.

```python
class TimeAccuracy(CallbacksBase):
   def __init__(self):
```
