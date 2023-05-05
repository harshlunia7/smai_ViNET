import yaml

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split

# pytorch lightning
from models.base_system import BaseSaliency
from models.base_modules import Decoder, S3D_Encoder
from evaluation.modules import Evaluation_Metric
import torchmetrics


class ViNet(BaseSaliency):
    def __init__(
        self, learning_rate, batch_size, use_upsample=True, num_hier=3, num_clips=32
    ):
        super().__init__(learning_rate=learning_rate)

        self.lr = learning_rate
        self.use_upsample = use_upsample
        self.num_hier = num_hier
        self.num_clips = num_clips
        self.batch_size = batch_size
        self.loss_module = Evaluation_Metric(batch_size=self.batch_size)
        # self.loss = torchmetrics.KLDivergence()

        self.decoder = Decoder(
            use_upsample=self.use_upsample,
            num_clips=self.num_clips,
            num_hier=self.num_hier,
        )
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

        img_clips = img_clips.permute((0, 2, 1, 3, 4))
        pred_sal = self.forward(img_clips)
        assert pred_sal.size() == gt_sal.size()
        loss = self.loss_module.compute_loss("KL_Divergence", pred_sal, gt_sal)
        # cc_loss = self.loss_module.compute_loss("CC", pred_sal, gt_sal)
        l1_norm = self.loss_module.compute_loss("L1", pred_sal, gt_sal)
        # AUROC_loss = self.loss_module.compute_loss("AUROC", pred_sal, gt_sal)
        # loss = self.loss(pred_sal.reshape((pred_sal.size(0), -1)), gt_sal.reshape((gt_sal.size(0), -1)))
        self.log_dict({"Loss": loss, "L1 Norm": l1_norm}, on_epoch=True)
        return loss
