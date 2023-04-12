import yaml
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split

# pytorch lightning
import pytorch_lightning as pl

class BaseSaliency(pl.LightningModule, ABC):
    def __init__(self, learning_rate):
        super().__init__()
        self.lr = learning_rate
    
    @abstractmethod
    def forward(self):
        raise NotImplementedError
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)