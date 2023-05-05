import yaml
import math

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split

# pytorch lightning
import pytorch_lightning as pl

from models.utils import *


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

    def forward(self, y0, y1, y2, y3):
        z = self.decoder_subblock_1(y0)
        if self.num_hier >= 1:
            z = torch.cat([z, y1], dim=2)
        z = self.decoder_subblock_2(z)
        if self.num_hier >= 2:
            z = torch.cat([z, y2], dim=2)
        z = self.decoder_subblock_3(z)
        if self.num_hier >= 3:
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


class SoundNet(nn.Module):
    def __init__(self):
        super(SoundNet, self).__init__()
  
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(64, 1), stride=(2, 1),
                               padding=(32, 0))
        self.batchnorm1 = nn.BatchNorm2d(16, eps=1e-5, momentum=0.1)
        self.relu1 = nn.ReLU(True)
        self.maxpool1 = nn.MaxPool2d((8, 1), stride=(8, 1))

        self.conv2 = nn.Conv2d(16, 32, kernel_size=(32, 1), stride=(2, 1),
                               padding=(16, 0))
        self.batchnorm2 = nn.BatchNorm2d(32, eps=1e-5, momentum=0.1)
        self.relu2 = nn.ReLU(True)
        self.maxpool2 = nn.MaxPool2d((8, 1), stride=(8, 1))

        self.conv3 = nn.Conv2d(32, 64, kernel_size=(16, 1), stride=(2, 1),
                               padding=(8, 0))
        self.batchnorm3 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1)
        self.relu3 = nn.ReLU(True)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=(8, 1), stride=(2, 1),
                               padding=(4, 0))
        self.batchnorm4 = nn.BatchNorm2d(128, eps=1e-5, momentum=0.1)
        self.relu4 = nn.ReLU(True)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=(4, 1), stride=(2, 1),
                               padding=(2, 0))
        self.batchnorm5 = nn.BatchNorm2d(256, eps=1e-5, momentum=0.1)
        self.relu5 = nn.ReLU(True)
        self.maxpool5 = nn.MaxPool2d((4, 1), stride=(4, 1))

        self.conv6 = nn.Conv2d(256, 512, kernel_size=(4, 1), stride=(2, 1),
                               padding=(2, 0))
        self.batchnorm6 = nn.BatchNorm2d(512, eps=1e-5, momentum=0.1)
        self.relu6 = nn.ReLU(True)

        self.conv7 = nn.Conv2d(512, 1024, kernel_size=(4, 1), stride=(2, 1),
                               padding=(2, 0))
        self.batchnorm7 = nn.BatchNorm2d(1024, eps=1e-5, momentum=0.1)
        self.relu7 = nn.ReLU(True)

    def forward(self, input_audio):
        audio_out = self.conv1(input_audio)
        audio_out = self.batchnorm1(audio_out)
        audio_out = self.relu1(audio_out)
        audio_out = self.maaudio_outpool1(audio_out)

        audio_out = self.conv2(audio_out)
        audio_out = self.batchnorm2(audio_out)
        audio_out = self.relu2(audio_out)
        audio_out = self.maaudio_outpool2(audio_out)

        audio_out = self.conv3(audio_out)
        audio_out = self.batchnorm3(audio_out)
        audio_out = self.relu3(audio_out)

        audio_out = self.conv4(audio_out)
        audio_out = self.batchnorm4(audio_out)
        audio_out = self.relu4(audio_out)

        audio_out = self.conv5(audio_out)
        audio_out = self.batchnorm5(audio_out)
        audio_out = self.relu5(audio_out)
        audio_out = self.maaudio_outpool5(audio_out)

        audio_out = self.conv6(audio_out)
        audio_out = self.batchnorm6(audio_out)
        audio_out = self.relu6(audio_out)

        audio_out = self.conv7(audio_out)
        audio_out = self.batchnorm7(audio_out)
        audio_out = self.relu7(audio_out)

        return audio_out

class PositionalEncoding(nn.Module):
    def __init__(self, feature_size, max_len=4):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, feature_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, feature_size, 2).float()
            * (-math.log(10000.0) / feature_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe
        return x


class Transformer(nn.Module):
    def __init__(
        self, feature_size, hidden_size=256, nhead=4, num_encoder_layers=3, max_len=4
    ):
        super(Transformer, self).__init__()
        self.pos_encoder = PositionalEncoding(feature_size, max_len=max_len)
        encoder_layers = nn.TransformerEncoderLayer(feature_size, nhead, hidden_size)

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_encoder_layers
        )

    def forward(self, embeddings):
        """embeddings: CxBxCh*H*W"""
        x = self.pos_encoder(embeddings)
        x = self.transformer_encoder(x)
        return x