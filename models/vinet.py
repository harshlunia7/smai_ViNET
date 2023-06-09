import torch
import torch.nn.functional as F

# pytorch lightning
from models.base_system import BaseSaliency
from models.base_modules import Decoder, S3D_Encoder
from evaluation.modules import Evaluation_Metric


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
        loss_dict = self._common_step(batch, batch_idx)
        self.log_dict(
            {
                "Train_Loss": loss_dict["Loss"],
                "Train_L1_Norm": loss_dict["L1_Norm"],
                "Train_cc_loss": loss_dict["cc_loss"],
                "Train_similarity": loss_dict["similarity"],
            }
        )
        return loss_dict["Loss"]

    def validation_step(self, batch, batch_idx):
        loss_dict = self._common_step(batch, batch_idx)
        self.log_dict(
            {
                "val_Loss": loss_dict["Loss"],
                "val_L1_Norm": loss_dict["L1_Norm"],
                "val_cc_loss": loss_dict["cc_loss"],
                "val_similarity": loss_dict["similarity"],
            }
        )

    def _common_step(self, batch, batch_idx):
        img_clips, gt_sal = batch

        img_clips = img_clips.permute((0, 2, 1, 3, 4))
        pred_sal = self.forward(img_clips)
        assert pred_sal.size() == gt_sal.size()

        loss = self.loss_module.compute_loss("KL_Divergence", pred_sal, gt_sal)
        l1_norm = self.loss_module.compute_loss("L1", pred_sal, gt_sal)
        similarity = self.loss_module.compute_loss("similarity", pred_sal, gt_sal)
        cc_loss = self.loss_module.compute_loss("CC", pred_sal, gt_sal)

        return {
            "Loss": loss,
            "L1_Norm": l1_norm,
            "cc_loss": cc_loss,
            "similarity": similarity,
        }
