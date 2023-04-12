import yaml

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split

# pytorch lightning
from models.base_system import BaseSaliency
from models.base_modules import Decoder, S3D_Encoder

class ViNet(BaseSaliency):
    def __init__(self, learning_rate, use_upsample=True, num_hier=3, num_clips=32):
        super().__init__(learning_rate=learning_rate)
        
        self.lr = learning_rate
        self.use_upsample = use_upsample
        self.num_hier = num_hier
        self.num_clips = num_clips
        self.loss_fn = None

        self.decoder = Decoder(use_upsample=self.use_upsample, num_clips=self.num_clips, num_hier=self.num_hier)
        self.backbone_encoder = S3D_Encoder()
    
    def forward(self, x):
        [y0, y1, y2, y3] = self.backbone_encoder(x)
        return self.decoder(y0, y1, y2, y3)
    
    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx)

    def _common_step(self, batch, batch_idx):
        img_clips, gt_sal = batch

        img_clips = img_clips.permute((0,2,1,3,4))
        pred_sal = self.forward(img_clips)
        assert pred_sal.size() == gt_sal.size()

        loss = self.loss_func(pred_sal, gt_sal)
        return loss