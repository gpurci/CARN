#  Callbacks

##  Description

**Objective**

Acest folder contine instrumente, de monitorizare, salvare sau executare comenzi externe, care sunt apelate in timpul atrenarii
---

## Project Structure

| File | Description |
|------|--------------|
| `callback_base.py` | Aceasta este clasa parinte, care contine toate metodele de baza, ce trebuie sa le detine un `callback`  |
| `callback_list.py` | Aceasta este clasa lista, care are ca parametru o lista de callback-uri, si le apeleaza in ordinea FIFO |
| `save_model_by_best_key.py` | Aceasta clasa salveaza un model, atunci cand se gaseste un parametru din loguri mai mare decat parametrul anterior |
| `save_logs2csv.py` | Aceassta clasa salveaza logurile intr-un fisier `csv`  |
| `save_fake_image.py` | Aceasta clasa este pentru generare de imagini, salveaza intr-un folder o lista de imagini de antrenament concatenate cu imagini sintetizate |
| `show_logs.py` | Aceasta clasa afiseaza cate un plot pentru fiecare log, (log-urile de antrenare si testare sunt afisate in acelasi plot) |
| `time_accuracy.py` | Aceasta clasa afiseaza timpul total de antrenament pana la epoca curenta si acuratetea datelor de validare |
| `test` | In acest folder sunt efectuate testele pentru `callback`-uri |

---

## callback_base

Aceasta este clasa parinte, care contine toate metodele de baza, ce trebuie sa le detine un `callback`, motodele neafisate sunt similare cu on_batch_begin, sunt abstracte, vezi mai mult in fisierul callback_base.py.

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

Aceasta este clasa lista, care are ca parametru o lista de callback-uri, si le apeleaza in ordinea FIFO, motodele neafisate sunt similare cu on_batch_begin, vezi mai mult in fisierul callback_list.py.
Unde:
- callbacks: lista de callback-uri ce mostenesc CallbacksBase.
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

Aceasta clasa salveaza un model, atunci cand se gaseste un parametru din loguri mai mare decat parametrul anterior, vezi mai mult in fisierul save_model_by_best_key.py.
Unde:
- model: type nn.Module; modelul care se doreste a fi salvat
- filename: type str; numele fisierului unde va fi salvat modelul
- key: type str; dupa care parametru va fi salvat fisierul
```python
class SaveModelByBestKey(CallbacksBase):
   def __init__(self, model, filename, key="val_acc"):
```

---

## save_logs2csv

Aceassta clasa salveaza logurile intr-un fisier `csv`, vezi mai mult in fisierul save_logs2csv.py.
Unde:
- filename: type str; numele fisierului unde var fi salvate logurile
```python
class SaveLogs2csv(CallbacksBase):
   def __init__(self, filename):
```

---

## save_fake_image

Aceasta clasa este pentru generare de imagini, salveaza intr-un folder o lista de imagini de antrenament concatenate cu imagini sintetizate, vezi mai mult in fisierul save_fake_image.py.
Unde:
- model: type nn.Module; modelul care se doreste a fi salvat
- dataset: type torch.utils.data.Dataset; obiectul de unde vom scoate datele, normalizate sau dupa cerintele modelului
- transform: type torchvision.transforms.v2.Transform; functia care va normaliza datele in intervalul 0...255
- device: type ; device-ul pe care va rula modelul
- size: type int; cate imagini vor fi scoarse din dataset
- path: type str; numele folderului unde vom salva imaginile
- freq: type int; freqventa cu care se va salva imaginile in 'path'
```python
class SaveFakeImage(CallbacksBase):
   def __init__(self, model, dataset, transform, device, size=1, path="", freq=1):
```

---

## show_logs

Aceasta clasa afiseaza cate un plot pentru fiecare log, (log-urile de antrenare si testare sunt afisate in acelasi plot), vezi mai mult in fisierul show_logs.py.
Unde:
- title: type str; titlul plotului
```python
class ShowLogs(CallbacksBase):
   def __init__(self, title):
```

---

## time_accuracy

Aceasta clasa afiseaza timpul total de antrenament pana la epoca curenta si acuratetea datelor de validare, vezi mai mult in fisierul time_accuracy.py.

```python
class TimeAccuracy(CallbacksBase):
   def __init__(self):
```
