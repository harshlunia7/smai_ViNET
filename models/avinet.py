import torch
import torch.nn.functional as F
from torch import nn

# pytorch lightning
from models.base_system import BaseSaliency
from models.base_modules import Transformer, SoundNet
from evaluation.modules import Evaluation_Metric

from models.vinet import ViNet


class AViNet(BaseSaliency):
    def __init__(
        self,
        learning_rate,
        batch_size,
        use_upsample=True,
        num_hier=3,
        num_clips=32,
        use_transformer=True,
        fusing_method="bilinear",
    ):
        super().__init__(learning_rate=learning_rate)

        self.lr = learning_rate
        self.use_upsample = use_upsample
        self.fusing_method = fusing_method
        self.num_hier = num_hier
        self.use_transformer = use_transformer
        self.num_clips = num_clips
        self.batch_size = batch_size
        self.loss_module = Evaluation_Metric(batch_size=self.batch_size)

        if self.fusing_method == "bilinear":
            self.transformer_in_channel = 336
            self.num_encoder_layers = 3
            self.nhead = 4
            self.transformer_max_length = 32
            self.conv_in_out_channel = 32
        elif self.fusing_method == "concat":
            self.transformer_in_channel = 512
            self.num_encoder_layers = 3
            self.nhead = 4
            self.transformer_max_length = 4*7*12+3
            self.conv_in_out_channel = 512

        self.visual_model = ViNet(
            use_upsample=self.use_upsample,
            num_hier=self.num_hier,
            num_clips=self.num_clips,
            batch_size=self.batch_size,
            learning_rate=self.lr,
        )

        self.conv_in_1x1 = nn.Conv3d(
            in_channels=1024,
            out_channels=self.conv_in_out_channel,
            kernel_size=1,
            stride=1,
            bias=True,
        )
        self.conv_out_1x1 = nn.Conv3d(
            in_channels=32, out_channels=1024, kernel_size=1, stride=1, bias=True
        )
        self.audio_conv_1x1 = nn.Conv2d(
            in_channels=1024,
            out_channels=self.conv_in_out_channel,
            kernel_size=1,
            stride=1,
            bias=True,
        )

        self.audio_encoder = SoundNet()

        self.transformer = Transformer(
            feature_size=self.transformer_in_channel,
            hidden_size=self.transformer_in_channel,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            max_len=self.transformer_max_length,
        )

        self.audio_encoder.load_state_dict(torch.load('./soundnet8_final.pth'))
        print("Loaded SoundNet Weights")
        for param in self.audio_encoder.parameters():
            param.requires_grad = True

        self.maxpool = nn.MaxPool3d((4, 1, 1), stride=(2, 1, 2), padding=(0, 0, 0))
        self.bilinear = nn.Bilinear(42, 3, 4 * 7 * 12)

    def _fuse_audio_video_features(self, audio_data, visual_features):
        if self.fusing_method == "concat":
            audio_data = self.audio_conv_1x1(audio_data)
            # HARSH SEE THE AUDIO SOUNDNET OUTPUT, SHAPE
            audio_data = audio_data.flatten(2)

            visual_features = self.conv_in_1x1(visual_features)
            visual_features = visual_features.flatten(2)
            # HARSH SEE THE y0 OUTPUT, SHAPE

            audio_visual_fused = torch.cat((visual_features, audio_data), 2)
            audio_visual_fused = audio_visual_fused.permute((2, 0, 1))
            # HARSH SEE THE audio_visual_fused OUTPUT, SHAPE and why permutation ??

            audio_visual_fused = self.transformer(audio_visual_fused)
            audio_visual_fused = audio_visual_fused.permute((1, 2, 0))
            # HARSH SEE THE audio_visual_fused transformer OUTPUT, SHAPE and why permutation ??

            video_features = audio_visual_fused[..., : 4 * 7 * 12]
            audio_features = audio_visual_fused[..., 4 * 7 * 12 :]

            video_features = video_features.view(
                video_features.size(0), video_features.size(1), 4, 7, 12
            )
            # HARSH:: averging over heads; **CHECK**
            audio_features = torch.mean(audio_features, dim=2)
            audio_features = audio_features.view(
                audio_features.size(0), audio_features.size(1), 1, 1, 1
            ).repeat(1, 1, 4, 7, 12)

            final_out = torch.cat((video_features, audio_features), 1)
        elif self.fusing_method == "bilinear":
            audio_data = audio_data.flatten(2)
            visual_features = self.maxpool(visual_features)
            visual_features = visual_features.flatten(2)
            audio_visual_fused = self.bilinear(visual_features, audio_data)
            final_out = audio_visual_fused.view(
                audio_visual_fused.size(0), audio_visual_fused.size(1), 4, 7, 12
            )

            if self.use_transformer:
                final_out = self.conv_in_1x1(final_out)
                final_out = final_out.flatten(2)
                final_out = final_out.permute((1, 0, 2))
                final_out = self.transformer(final_out)
                final_out = final_out.permute((1, 0, 2))
                final_out = final_out.view(
                    final_out.size(0), final_out.size(1), 4, 7, 12
                )
                final_out = self.conv_out_1x1(final_out)

        return final_out

    def forward(self, x, audio_data):
        audio_data = self.audio_encoder(audio_data)
        [y0, y1, y2, y3] = self.visual_model.backbone_encoder(x)

        final_out = self._fuse_audio_video_features(audio_data, y0)

        return self.visual_model.decoder(final_out, y1, y2, y3)

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx)

    def _common_step(self, batch, batch_idx):
        img_clips, gt_sal, audio_features = batch

        img_clips = img_clips.permute((0, 2, 1, 3, 4))
        pred_sal = self.forward(img_clips, audio_features)

        assert pred_sal.size() == gt_sal.size()

        loss = self.loss_module.compute_loss("KL_Divergence", pred_sal, gt_sal)
        l1_norm = self.loss_module.compute_loss("L1", pred_sal, gt_sal)
        similarity = self.loss_module.compute_loss("similarity", pred_sal, gt_sal)
        cc_loss = self.loss_module.compute_loss("CC", pred_sal, gt_sal)

        self.log_dict(
            {
                "Loss": loss,
                "L1 Norm": l1_norm,
                "cc_loss": cc_loss,
                "similarity": similarity,
            }
        )
        return loss
