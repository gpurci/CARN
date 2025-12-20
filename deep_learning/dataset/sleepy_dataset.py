#!/usr/bin/python

import torch
from torch.utils.data import Dataset, DataLoader
from timed_decorator.simple_timed import timed

class SleepyDataset(Dataset):
    def __init__(self):
        self.data = [1, 2, 3, 4]
        self.labels = ["odd", "even", "odd", "even"]
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i: int):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            print(f"Loading item {i} in main process")
        else:
            print(f"Worker {worker_info.id}/{worker_info.num_workers} is loading item {i}")

        time.sleep(1)  # Simulate a slow loading process
        return self.data[i], self.labels[i]
    
