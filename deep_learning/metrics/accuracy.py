#!/usr/bin/python

class Acuracy(Metrics):
   def __init__(self, **kw):
      super().__init__(**kw)
      pass

   def __call__(self, y_pred, y):
      # y_pred: logits or probabilities (batch_size, num_classes)
      # y_true: true labels (batch_size,)
      preds = torch.argmax(y_pred, dim=1)
      y     = torch.argmax(y,      dim=1)
      correct = (preds == y).sum().item()
      total = y.size(0)
      return correct / total
