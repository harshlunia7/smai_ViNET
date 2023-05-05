import sys
import os

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split

from torchvision import transforms, utils

from PIL import Image
import numpy as np
import cv2


class DHF1KDataset(Dataset):
    def __init__(
        self,
        data_directory,
        clip_length,
        dataset_type="train",
        multi_frame=0,
        clip_step=1,
    ):
        super().__init__()
        self.data_path = data_directory
        self.clip_length = clip_length
        self.dataset_type = dataset_type
        self.multi_frame = multi_frame
        self.clip_step = clip_step
        self.img_transform = transforms.Compose(
            [
                transforms.Resize((224, 384)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        # Get the video names
        self.video_names = os.listdir(self.data_path)

        if self.dataset_type == "train":
            # Get the number of frames for every video
            self.num_frames_per_video = [
                len(os.listdir(os.path.join(self.data_path, video_name, "images")))
                for video_name in self.video_names
            ]
        elif self.dataset_type == "val":
            self.num_frames_per_video = []
            for video_name in self.video_names:
                for i in range(
                    0,
                    len(os.listdir(os.path.join(self.data_path, video_name, "images")))
                    - (self.clip_step * self.clip_length),
                    4
                    * self.clip_length,  # Varun:: Why is 4 multiplied for step size ??
                ):
                    self.num_frames_per_video.append((video_name, i))
        else:
            self.num_frames_per_video = []
            for video_name in self.video_names:
                for i in range(
                    0,
                    len(os.listdir(os.path.join(self.data_path, video_name, "images")))
                    - (self.clip_step * self.clip_length),
                    self.clip_length,
                ):
                    self.num_frames_per_video.append((video_name, i))
                # Varun :: Why is the last possible clip added explicitly ?
                self.num_frames_per_video.append(
                    (
                        video_name,
                        len(
                            os.listdir(
                                os.path.join(self.data_path, video_name, "images")
                            )
                        )
                        - self.clip_length,
                    )
                )

    def __len__(self):
        return len(self.num_frames_per_video)

    def __getitem__(self, idx):
        if self.dataset_type == "train":
            video_name = self.video_names[idx]
            start_idx = np.random.randint(
                0,
                self.num_frames_per_video[idx]
                - (self.clip_step * self.clip_length)
                + 1,
            )
        elif self.dataset_type == "val":
            (video_name, start_idx) = self.num_frames_per_video[idx]

        video_frames = os.path.join(self.data_path, video_name, "images")
        video_maps = os.path.join(self.data_path, video_name, "maps")

        clip_img = []
        clip_gt = []

        for i in range(self.clip_length):
            img = Image.open(
                os.path.join(
                    video_frames, "%04d.png" % (start_idx + (self.clip_step * i) + 1)
                )
            ).convert("RGB")

            gt = np.array(
                Image.open(
                    os.path.join(
                        video_maps, "%04d.png" % (start_idx + (self.clip_step * i) + 1)
                    )
                ).convert("L")
            )
            gt = gt.astype("float")

            # if self.dataset_type == "train":
            gt = cv2.resize(gt, (384, 224))
            # gt = gt.resize((384, 224))

            if np.max(gt) > 1.0:
                gt = gt / 255.0
            clip_gt.append(torch.FloatTensor(gt))

            clip_img.append(self.img_transform(img))

        clip_img = torch.FloatTensor(torch.stack(clip_img, dim=0))
        clip_gt = torch.FloatTensor(torch.stack(clip_gt, dim=0))

        if self.multi_frame == 0:
            return clip_img, clip_gt[-1]
        return clip_img, clip_gt
