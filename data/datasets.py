import sys
import os
import math

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split

from torchvision import transforms, utils
import torchaudio

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


class AudioVideoDataset(Dataset):
    def __init__(
        self,
        data_directory,
        clip_length,
        dataset_name="DIEM",
        dataset_type="train",
        data_split=-1,
        use_sound=True,
        add_noise=False,
    ):
        super().__init__()
        self.data_path = data_directory
        self.dataset_name = dataset_name
        self.clip_length = clip_length
        self.dataset_type = dataset_type
        self.use_sound = use_sound
        self.add_noise = add_noise
        self.data_split = data_split
        self.max_audio_sampling_rate = 22050
        self.min_video_fps = 10
        self.video_names = []
        self.max_audio_win = int(self.max_audio_sampling_rate / self.min_video_fps * 32)
        self.img_transform = transforms.Compose(
            [
                transforms.Resize((224, 384)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        if self.data_split == -1:
            self.data_record_filename = (
                f"{self.dataset_name}_list_{self.dataset_type}_fps.txt"
            )
        else:
            self.data_record_filename = f"{self.dataset_name}_list_{self.dataset_type}_{self.data_split}_fps.txt"
        if self.dataset_type == "val":
            self.data_record_filename = self.data_record_filename.replace("val", "test")
        # Get the video names
        with open(
            os.path.join(self.data_path, self.dataset_name, self.data_record_filename),
            "r",
        ) as file_obj:
            for entry in file_obj.readlines():
                self.video_names.append(entry.strip().split(" ")[0])
        self.video_names.sort()

        if self.dataset_type == "train":
            # Get the number of frames for every video
            self.num_frames_per_video = [
                len(
                    os.listdir(
                        os.path.join(
                            self.data_path,
                            self.dataset_name,
                            "annotations",
                            video_name,
                            "maps",
                        )
                    )
                )
                for video_name in self.video_names
            ]
        elif self.dataset_type == "val" or self.dataset_type == "test":
            self.num_frames_per_video = []
            for video_name in self.video_names:
                number_frames = os.listdir(
                    os.path.join(
                        self.data_path,
                        self.dataset_name,
                        "annotations",
                        video_name,
                        "maps",
                    )
                )
                for i in range(
                    0,
                    len(number_frames) - self.clip_length,
                    2 * self.clip_length,
                ):
                    if self._check_frame(
                        os.path.join(
                            self.data_path,
                            self.dataset_name,
                            "annotations",
                            video_name,
                            "maps",
                            f"eyeMap_{str(i+self.clip_length).zfill(5)}.jpg",
                        )
                    ):
                        self.num_frames_per_video.append((video_name, i))

        if self.use_sound:
            if self.dataset_type == "val":
                self.data_record_filename = self.data_record_filename.replace(
                    "val", "test"
                )
            if self.add_noise == False:
                print("Note:: Using Original Audio Files")
                audio_or_noise_path = os.path.join(
                    self.data_path, self.dataset_name, "video_audio"
                )
            else:
                print("Note:: Using Noise Files")
                audio_or_noise_path = os.path.join(
                    self.data_path, self.dataset_name, "video_noise"
                )
            self.audio_metadata = self._collect_video_audio_metadata(
                data_list_path=os.path.join(
                    self.data_path, self.dataset_name, self.data_record_filename
                ),
                audio_path=audio_or_noise_path,
                annotation_path=os.path.join(
                    self.data_path, self.dataset_name, "annotations"
                ),
            )

    def _check_frame(self, last_frame_path):
        return cv2.imread(last_frame_path, 0).max != 0

    def _collect_video_audio_metadata(
        self, data_list_path, audio_path, annotation_path
    ):
        video_names = []
        video_fps = []
        with open(data_list_path, "r") as file_obj:
            for entry in file_obj.readlines():
                params = entry.strip().split(" ")
                video_names.append(params[0])
                video_fps.append(float(params[2]))

        audio_metadata = {}
        for i in range(len(video_names)):
            if i % 100 == 0:
                print(f"dataset loading [{i}/{len(video_names)}]")

            video_total_frames = len(
                os.listdir(os.path.join(annotation_path, video_names[i], "maps"))
            )
            if video_total_frames <= 1:
                print("Insufficient frames")
                continue

            audio_wav_path = os.path.join(
                audio_path, video_names[i], video_names[i] + ".wav"
            )
            if not os.path.exists(audio_wav_path):
                print(f"Audio file {audio_wav_path} does not exist!")
                continue

            [audiowav, audio_sample_rate] = torchaudio.load(audio_wav_path)
            audiowav = audiowav * (2**-23)

            audio_sample_per_video_frame = audio_sample_rate / float(video_fps[i])
            audio_video_sync_buffer = audio_sample_per_video_frame / 2
            audio_total_samples = audiowav.shape[1]
            starts = np.zeros(video_total_frames + 1, dtype=int)
            ends = np.zeros(video_total_frames + 1, dtype=int)
            starts[0] = 0
            ends[0] = 0
            for videoframe in range(1, video_total_frames + 1):
                starts[videoframe] = int(
                    max(
                        0,
                        (
                            (videoframe - 1)
                            * (1.0 / float(video_fps[i]))
                            * audio_sample_rate
                        )
                        - audio_video_sync_buffer,
                    )
                )
                ends[videoframe] = int(
                    min(
                        audio_total_samples,
                        abs(
                            (
                                (videoframe - 1)
                                * (1.0 / float(video_fps[i]))
                                * audio_sample_rate
                            )
                            + audio_video_sync_buffer
                        ),
                    )
                )

            audioinfo = {
                "audiopath": audio_path,
                "video_id": video_names[i],
                "Fs": audio_sample_rate,
                "wav": audiowav,
                "starts": starts,
                "ends": ends,
            }

            audio_metadata[video_names[i]] = audioinfo

        return audio_metadata

    def _get_audio_feature(self, video_name, audio_metadata, clip_length, start_idx):
        audioexcer = torch.zeros(1, self.max_audio_win)
        audioexcerpt_total_samples = 0

        if video_name in audio_metadata:
            video_audio_data = audio_metadata[video_name]
            excerptstart = video_audio_data["starts"][start_idx + 1]

            if (start_idx + clip_length) >= len(video_audio_data["ends"]):
                print("Exceeds size", video_name)
                sys.stdout.flush()
                excerptend = video_audio_data["ends"][-1]
            else:
                excerptend = video_audio_data["ends"][start_idx + clip_length]
            try:
                audioexcerpt_total_samples = video_audio_data["wav"][
                    :, excerptstart : excerptend + 1
                ].shape[1]
                audioexcer[
                    :,
                    ((audioexcer.shape[1] // 2) - (audioexcerpt_total_samples // 2)) : (
                        (audioexcer.shape[1] // 2)
                        + (audioexcerpt_total_samples // 2)
                        + ((audioexcerpt_total_samples % 2) * 1)
                    ),
                ] = (
                    torch.from_numpy(
                        np.hanning(
                            video_audio_data["wav"][
                                :, excerptstart : excerptend + 1
                            ].shape[1]
                        )
                    ).float()
                    * video_audio_data["wav"][:, excerptstart : excerptend + 1]
                )
            except:
                print(
                    f"Not able to get the audio excerpt from {excerptstart} to {excerptend} in {video_name}"
                )

        else:
            print(video_name, "not present in data")
        audio_feature = audioexcer.view(1, -1, 1)
        return audio_feature

    def __len__(self):
        return len(self.num_frames_per_video)

    def __getitem__(self, idx):
        if self.dataset_type == "train":
            video_name = self.video_names[idx]
            while True:
                start_idx = np.random.randint(
                    0, self.num_frames_per_video[idx] - self.clip_length + 1
                )
                if self._check_frame(
                    os.path.join(
                        self.data_path,
                        self.dataset_name,
                        "annotations",
                        video_name,
                        "maps",
                        f"eyeMap_{str(start_idx+self.clip_length).zfill(5)}.jpg",
                    )
                ):
                    break
                else:
                    print("No saliency defined in train dataset")
                    sys.stdout.flush()

        elif self.dataset_type == "test" or self.dataset_type == "val":
            (video_name, start_idx) = self.num_frames_per_video[idx]

        clip_path = os.path.join(
            self.data_path, self.dataset_name, "video_frames", video_name
        )
        annotation_path = os.path.join(
            self.data_path, self.dataset_name, "annotations", video_name, "maps"
        )

        if self.use_sound:
            audio_feature = self._get_audio_feature(
                video_name, self.audio_metadata, self.clip_length, start_idx
            )

        clip_img = []

        for i in range(self.clip_length):
            img = Image.open(
                os.path.join(clip_path, f"img_{str(start_idx+i+1).zfill(5)}.jpg")
            ).convert("RGB")
            clip_img.append(self.img_transform(img))

        clip_img = torch.FloatTensor(torch.stack(clip_img, dim=0))

        gt = np.array(
            Image.open(
                os.path.join(
                    annotation_path,
                    f"eyeMap_{str(start_idx+self.clip_length).zfill(5)}.jpg",
                )
            ).convert("L")
        )
        gt = gt.astype("float")

        # if self.dataset_type == "train":
        gt = cv2.resize(gt, (384, 224))

        if np.max(gt) > 1.0:
            gt = gt / 255.0
        assert gt.max() != 0, (str(start_idx + self.clip_length).zfill(5), video_name)
        # try:
        #     assert gt.max() != 0, (str(start_idx+self.clip_length).zfill(5), video_name)
        # except:
        #     # gt += 2 * math.exp(-10)

        # print(str(start_idx+self.clip_length).zfill(5), video_name)
        if self.use_sound:
            return clip_img, gt, audio_feature
        return clip_img, gt
