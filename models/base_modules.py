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
		super(S3D_Encoder, self).__init__()
		
		self.base1 = nn.Sequential(
			SepConv3d(3, 64, kernel_size=7, stride=2, padding=3),
			nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
			BasicConv3d(64, 64, kernel_size=1, stride=1),
			SepConv3d(64, 192, kernel_size=3, stride=1, padding=1),
		)
		self.maxp2 = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
		self.base2 = nn.Sequential(
			Mixed_3b(),
			Mixed_3c(),
		)
		self.maxp3 = nn.MaxPool3d(kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1))
		self.base3 = nn.Sequential(
			Mixed_4b(),
			Mixed_4c(),
			Mixed_4d(),
			Mixed_4e(),
			Mixed_4f(),
		)
		self.maxt4 = nn.MaxPool3d(kernel_size=(2,1,1), stride=(2,1,1), padding=(0,0,0))
		self.maxp4 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2), padding=(0,0,0))
		self.base4 = nn.Sequential(
			Mixed_5b(),
			Mixed_5c(),
		)

	def forward(self, x):
		# print('input', x.shape)
		y3 = self.base1(x)
		# print('base1', y3.shape)
		
		y = self.maxp2(y3)
		# print('maxp2', y.shape)

		y2 = self.base2(y)
		# print('base2', y2.shape)

		y = self.maxp3(y2)
		# print('maxp3', y.shape)

		y1 = self.base3(y)
		# print('base3', y1.shape)

		y = self.maxt4(y1)
		y = self.maxp4(y)
		# print('maxt4p4', y.shape)

		y0 = self.base4(y)

		return [y0, y1, y2, y3]