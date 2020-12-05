# Importing Libraries
import os
import itertools
import yaml
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser, Namespace

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from models import build_model
from dataloaders import build_mnist

def main(hparams):
    # seeding
    pl.seed_everything(hparams.seed)
    
    # DataLoaders
    transform=transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))])
    
    train_dataloader, val_dataloader, test_dataloader = build_mnist(hparams.val_ratio, hparams.batch_size, hparams.num_workers, transform)
    
    # System
    model = build_model(hparams)
    
    # Logging
    if "tensorboard_logs" in os.listdir() and \
        hparams.experiment_name in os.listdir("tensorboard_logs") and \
        len(os.listdir(f"tensorboard_logs/{hparams.experiment_name}"))>0:
        versions = os.listdir(f"tensorboard_logs/{hparams.experiment_name}")
        version_nums = []
        for version in versions:
            if 'version' in version:
                version_nums.append(int(version.split('_')[-1]))
        if len(version_nums)==0:
            version_nums.append(-1)
        version_num = max(version_nums) + 1
    else:
        version_num = 0
    hparams = vars(hparams)
    hparams['version_num'] =  version_num
    hparams = Namespace(**hparams)
    logger = TensorBoardLogger(
        save_dir=os.path.join(os.getcwd(), "tensorboard_logs"), \
        version = version_num,
        name=hparams.experiment_name)
    print(hparams)
    
    # Checkpoint saving
    # dirpath=f'tensorboard_logs/{args.experiment_name}/version_{args.version_num}/checkpoints',
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename='{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min')
    
    # Trainer
    trainer = pl.Trainer(gpus=hparams.gpus, max_epochs=hparams.epochs, progress_bar_refresh_rate=10, \
                         profiler=True, logger=logger, callbacks=[checkpoint_callback])
    
    # Training
    trainer.fit(model, train_dataloader, val_dataloader)
    
    # Saving Configuration
    yaml_path = f'tensorboard_logs/{args.experiment_name}/version_{args.version_num}'
    with open(f'{yaml_path}/conf.yml', 'w') as outfile:
        yaml.dump(vars(args), outfile, default_flow_style=False)
    
    # Testing
    trainer.test(test_dataloaders=test_dataloader)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpus', type=int, default=-1)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--experiment_name', type=str)
    args = parser.parse_args()

    main(args)