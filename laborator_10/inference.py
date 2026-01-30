#!/usr/bin/python

import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as 

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', type=str, default='./sample_data')
    p.add_argument('--image_size', type=int, default=256)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--latent_channels', type=int, default=128)
    p.add_argument('--beta_kl', type=float, default=1e-3)
    p.add_argument('--out', type=str, default='./checkpoints/vae.pt')
    return p.parse_args()


def main():
    args = parse_args()




if __name__ == '__main__':
    main()