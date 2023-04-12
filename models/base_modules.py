import yaml

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split

# pytorch lightning
import pytorch_lightning as pl


class Decoder(nn.Module):
    def __init__(self, use_upsample=True, num_hier=3, num_clips=32):
        super().__init__()
        self.num_hier = num_hier
        self.num_clips = num_clips
        self.use_upsample = use_upsample

        self.decoder_config = self._get_config()

        self.upsampling = nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear")
        self.decoder_subblock_1 = nn.Sequential(
            nn.Conv3d(
                self.decoder_config["SUBBLOCK_1"]["IN_CHANNEL"],
                self.decoder_config["SUBBLOCK_1"]["OUT_CHANNEL"],
                kernel_size=self.decoder_config["SUBBLOCK_1"]["KERNEL_SIZE"],
                stride=self.decoder_config["SUBBLOCK_1"]["STRIDE"],
                padding=self.decoder_config["SUBBLOCK_1"]["PADDING"],
                bias=self.decoder_config["SUBBLOCK_1"]["BIAS"],
            ),
            nn.ReLU(),
            self.upsampling,
        )
        self.decoder_subblock_2 = nn.Sequential(
            nn.Conv3d(
                self.decoder_config["SUBBLOCK_2"]["IN_CHANNEL"],
                self.decoder_config["SUBBLOCK_2"]["OUT_CHANNEL"],
                kernel_size=self.decoder_config["SUBBLOCK_2"]["KERNEL_SIZE"],
                stride=self.decoder_config["SUBBLOCK_2"]["STRIDE"],
                padding=self.decoder_config["SUBBLOCK_2"]["PADDING"],
                bias=self.decoder_config["SUBBLOCK_2"]["BIAS"],
            ),
            nn.ReLU(),
            self.upsampling,
        )
        self.decoder_subblock_3 = nn.Sequential(
            nn.Conv3d(
                self.decoder_config["SUBBLOCK_3"]["IN_CHANNEL"],
                self.decoder_config["SUBBLOCK_3"]["OUT_CHANNEL"],
                kernel_size=self.decoder_config["SUBBLOCK_3"]["KERNEL_SIZE"],
                stride=self.decoder_config["SUBBLOCK_3"]["STRIDE"],
                padding=self.decoder_config["SUBBLOCK_3"]["PADDING"],
                bias=self.decoder_config["SUBBLOCK_3"]["BIAS"],
            ),
            nn.ReLU(),
            self.upsampling,
        )
        self.decoder_subblock_4 = nn.Sequential(
            nn.Conv3d(
                self.decoder_config["SUBBLOCK_4"]["IN_CHANNEL"],
                self.decoder_config["SUBBLOCK_4"]["OUT_CHANNEL"],
                kernel_size=self.decoder_config["SUBBLOCK_4"]["KERNEL_SIZE"],
                stride=self.decoder_config["SUBBLOCK_4"]["STRIDE"],
                padding=self.decoder_config["SUBBLOCK_4"]["PADDING"],
                bias=self.decoder_config["SUBBLOCK_4"]["BIAS"],
            ),
            nn.ReLU(),
            self.upsampling,  # 112 x 192
        )
        self.decoder_subblock_5 = nn.Sequential(
            nn.Conv3d(
                self.decoder_config["SUBBLOCK_5"]["IN_CHANNEL"],
                self.decoder_config["SUBBLOCK_5"]["OUT_CHANNEL"],
                kernel_size=self.decoder_config["SUBBLOCK_5"]["KERNEL_SIZE"],
                stride=self.decoder_config["SUBBLOCK_5"]["STRIDE"],
                padding=self.decoder_config["SUBBLOCK_5"]["PADDING"],
                bias=self.decoder_config["SUBBLOCK_5"]["BIAS"],
            ),
            nn.ReLU(),
            self.upsampling,  # 224 x 384
        )
        if self.num_hier != 3 or (
            self.num_hier == 3 and (self.num_clips != 8 and self.num_clips != 16)
        ):
            self.decoder_subblock_6 = nn.Sequential(
                nn.Conv3d(
                    self.decoder_config["SUBBLOCK_6"]["IN_CHANNEL"],
                    self.decoder_config["SUBBLOCK_6"]["OUT_CHANNEL"],
                    kernel_size=self.decoder_config["SUBBLOCK_6"]["KERNEL_SIZE"],
                    stride=self.decoder_config["SUBBLOCK_6"]["STRIDE"],
                    padding=self.decoder_config["SUBBLOCK_6"]["PADDING"],
                    bias=self.decoder_config["SUBBLOCK_6"]["BIAS"],
                ),
                nn.ReLU(),
            )
        self.decoder_subblock_7 = nn.Sequential(
            nn.Conv3d(
                self.decoder_config["SUBBLOCK_7"]["IN_CHANNEL"],
                self.decoder_config["SUBBLOCK_7"]["OUT_CHANNEL"],
                kernel_size=self.decoder_config["SUBBLOCK_7"]["KERNEL_SIZE"],
                stride=self.decoder_config["SUBBLOCK_7"]["STRIDE"],
                padding=self.decoder_config["SUBBLOCK_7"]["PADDING"],
                bias=self.decoder_config["SUBBLOCK_7"]["BIAS"],
            ),
            nn.Sigmoid(),
        )

    def _get_config(self):
        with open("/home2/rafaelgetto/smai_ViNET/models/decoder_config.yaml") as file:
            yaml_data = yaml.safe_load(file)
        if self.num_hier == 0:
            dc = yaml_data["NO_HIERARCHY"]
        elif self.num_hier == 1:
            dc = yaml_data["ONE_HIERARCHY"]
        elif self.num_hier == 2:
            dc = yaml_data["TWO_HIERARCHY"]
        elif self.num_hier == 3:
            if self.num_clips == 8:
                dc = yaml_data["CLIP_SIZE_8"]
            elif self.num_clips == 16:
                dc = yaml_data["CLIP_SIZE_16"]
            elif self.num_clips == 32:
                dc = yaml_data["CLIP_SIZE_32"]
            elif self.num_clips == 48:
                dc = yaml_data["CLIP_SIZE_48"]
        return dc

    def forward(self, y0, y1=torch.Tensor(), y2=torch.Tensor(), y3=torch.Tensor()):
        z = self.decoder_subblock_1(y0)
        z = torch.cat([z, y1], dim=2)
        z = self.decoder_subblock_2(z)
        z = torch.cat([z, y2], dim=2)
        z = self.decoder_subblock_3(z)
        z = torch.cat([z, y3], dim=2)
        z = self.decoder_subblock_4(z)
        z = self.decoder_subblock_5(z)
        if self.num_hier != 3 or (
            self.num_hier == 3 and (self.num_clips != 8 and self.num_clips != 16)
        ):
            z = self.decoder_subblock_6(z)
        z = self.decoder_subblock_7(z)
        z = z.view(z.size(0), z.size(3), z.size(4))
        return z

class S3D_Encoder(nn.Module):
    """
    The forward method of the encoder should return 4 feature tensors y0, y1, y2, y3.
    """
    def __init__(self):
        super().__init__()