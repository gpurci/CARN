#!/usr/bin/python

from torch.utils.data import DataLoader
from timed_decorator.simple_timed import timed

@timed(use_seconds=True, show_args=True, return_time=True)
def time_mon_load_data(dataset, num_workers, batch_size=1, shuffle=True, drop_last=False):
   dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)
   for _ in dataloader:
      pass  # Simulate training