# Importing Libraries
import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from models import CNN3
from dataloaders import build_mnist

def main(hparams):
    # seeding
    pl.seed_everything(hparams.seed)
    
    # DataLoaders
    transform=transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))])
    
    train_dataloader, val_dataloader, test_dataloader = build_mnist(hparams.val_ratio, hparams.batch_size, transform)
    
    # System
    model = CNN3()
    
    # Trainer
    logger = TensorBoardLogger(
    save_dir=os.path.join(os.getcwd(), "tensorboard_logs"),
    # version=1,
    name='CNN3'
    )
    trainer = pl.Trainer(gpus=hparams.gpus, max_epochs=hparams.epochs, progress_bar_refresh_rate=10, \
                         profiler=True, logger=logger)
    trainer.fit(model, train_dataloader, val_dataloader)
    
    # Testing
    trainer.test(test_dataloaders=test_dataloader)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpus', type=int, default=-1)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    main(args)