import os

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch.utils.data import random_split

def build_mnist(val_ratio, batch_size, transform):
    train_dataset = MNIST(os.path.join(os.getcwd(),"data/"), train=True, download=True, transform=transform)
    test_dataset = MNIST(os.path.join(os.getcwd(),"data/"), train=False, download=True, transform=transform)

    train, val = random_split(train_dataset, [int(train_dataset.__len__()*(1-val_ratio)), int(train_dataset.__len__()*val_ratio)])
    train = DataLoader(train, num_workers=8, batch_size=batch_size)
    val = DataLoader(val, num_workers=8, batch_size=batch_size)

    test = DataLoader(test_dataset, num_workers=8, batch_size=batch_size)
    
    return train, val, test