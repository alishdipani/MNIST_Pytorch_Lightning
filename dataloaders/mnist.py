import os

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch.utils.data import random_split

def build_mnist(val_ratio, batch_size, num_workers, transform):
    train_dataset = MNIST(os.path.join(os.getcwd(),"data/"), train=True, download=True, transform=transform)
    test_dataset = MNIST(os.path.join(os.getcwd(),"data/"), train=False, download=True, transform=transform)
    
    val_size = int(train_dataset.__len__()*val_ratio) # size of validation data
    train, val = random_split(train_dataset, [train_dataset.__len__()-val_size, val_size])
    train = DataLoader(train, num_workers=num_workers, batch_size=batch_size)
    val = DataLoader(val, num_workers=num_workers, batch_size=batch_size)

    test = DataLoader(test_dataset, num_workers=num_workers, batch_size=batch_size)
    
    return train, val, test