import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.metrics.functional.classification import confusion_matrix

from utils.metrics import plot_confusion_matrix

class CNN3(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # 28*28*1
        self.CNN = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=8,kernel_size=5,stride=1,padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2), # 14*14*8
            nn.Conv2d(in_channels=8,out_channels=64,kernel_size=5,stride=1,padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # 7*7*64
            nn.Conv2d(in_channels=64,out_channels=256,kernel_size=5,stride=1,padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2) # 3*3*256
        )       
        self.linear = nn.Linear(3*3*256,10)
        
    def forward(self, x):
        x = self.CNN(x)
        x = x.view(x.size(0),-1)
        x = self.linear(x)
        output = F.softmax(x, dim=1) 
        return output
    
    def training_step(self, batch, batch_idx):
        # getting outputs
        data, target = batch
        h = self.CNN(data)
        y_hat = F.log_softmax(self.linear(h.view(h.size(0),-1)), dim=1)
        # metrics
        acc = accuracy(y_hat, target=target)
        loss = F.nll_loss(y_hat, target)
        # logging
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_accuracy', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # getting outputs
        data, target = batch
        h = self.CNN(data)
        y_hat = F.log_softmax(self.linear(h.view(h.size(0),-1)), dim=1)
        # metrics
        acc = accuracy(y_hat, target=target)
        loss = F.nll_loss(y_hat, target)
        # logging
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def on_test_epoch_start(self):
        self.test_pred = None # test predicitons
        return None
    
    def test_step(self, batch, batch_idx):
        # getting outputs
        data, target = batch
        h = self.CNN(data)
        y_hat = F.log_softmax(self.linear(h.view(h.size(0),-1)), dim=1)
        # storing predictions for confusion matrix
        if self.test_pred is None:
            self.test_pred = torch.cat((y_hat.argmax(dim=1).unsqueeze(1),\
                                        target.unsqueeze(1)),dim=1)
        else:
            self.test_pred = torch.cat( (self.test_pred,\
                                         torch.cat((y_hat.argmax(dim=1).unsqueeze(1),\
                                                    target.unsqueeze(1)),dim=1)),\
                                       dim=0 )
        # metrics
        acc = accuracy(y_hat, target=target)
        loss = F.nll_loss(y_hat, target)
        # logging
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_accuracy', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_test_epoch_end(self):
        # confusion matrix
        cm = confusion_matrix(self.test_pred[:,0], self.test_pred[:,1],\
                              normalize=True, num_classes=10)
        plot_confusion_matrix(cm.cpu().numpy(), \
                              target_names=[str(i) for i in range(10)], \
                              title='Confusion Matrix',normalize=False, \
                              cmap=plt.get_cmap('bwr'))
        return None

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer