#!/usr/bin/python

class ShowLogs():
   def __init__(self, logging_var):
      self.logging_var = logging_var

   def print_columns(self, columns_list, is_head=False, is_final_entry=False):
      print_string = ""
      for col in columns_list:
         print_string += "|  %s  " % col
      print_string += "|"
      if is_head:
         print("-"*len(print_string))
      print(print_string)
      if is_head or is_final_entry:
         print("-"*len(print_string))

   logging_columns_list = ["run   ", "epoch", "train_acc", "val_acc", "tta_val_acc", "time_seconds"]
   def show(self, variables, is_final_entry):
      formatted = []
      for col in self.logging_var:
         var = variables.get(col.strip(), None)
         if type(var) in (int, str):
            res = str(var)
         elif type(var) is float:
            res = "{:0.4f}".format(var)
         else:
            assert var is None
            res = ""
         formatted.append(res.rjust(len(col)))
      self.print_columns(formatted, is_final_entry=is_final_entry)
