import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms as T

from data.datasets import DHF1KDataset, AudioVideoDataset


class SaliencyDataModule(pl.LightningDataModule):
    VISUAL_DATASET = ["DHF1K", "DIEM"]

    def __init__(
        self,
        dataset_name,
        root_data_dir,
        clip_length,
        multi_frame=0,
        split=-1,
        clip_step=1,
        use_sound=True,
        add_noise=False,
        batch_size=32,
        num_workers=2,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        # root_data_dir will contain train, val and test folders
        self.root_data_dir = root_data_dir
        self.clip_length = clip_length
        self.multi_frame = multi_frame
        self.split = split
        self.use_sound = use_sound
        self.add_noise = add_noise
        self.clip_step = clip_step
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

    @property
    def train_dataset(self):
        if (self._train_dataset == None) and (self.dataset_name in self.VISUAL_DATASET):
            if self.dataset_name == self.VISUAL_DATASET[0]:
                self._train_dataset = DHF1KDataset(
                    data_directory=os.path.join(self.root_data_dir, "train"),
                    clip_length=self.clip_length,
                    multi_frame=self.multi_frame,
                    dataset_type="train",
                    clip_step=self.clip_step,
                )
            elif self.dataset_name == self.VISUAL_DATASET[1]:
                self._train_dataset = AudioVideoDataset(
                    data_directory=self.root_data_dir,
                    clip_length=self.clip_length,
                    dataset_name=self.VISUAL_DATASET[1],
                    dataset_type="train",
                    data_split=self.split,
                    use_sound=self.use_sound,
                    add_noise=self.add_noise,
                )
        return self._train_dataset

    @property
    def val_dataset(self):
        if (self._val_dataset == None) and (self.dataset_name in self.VISUAL_DATASET):
            if self.dataset_name == self.VISUAL_DATASET[0]:
                self._val_dataset = DHF1KDataset(
                    data_directory=os.path.join(self.root_data_dir, "val"),
                    clip_length=self.clip_length,
                    multi_frame=self.multi_frame,
                    dataset_type="val",
                    clip_step=self.clip_step,
                )
            elif self.dataset_name == self.VISUAL_DATASET[1]:
                self._val_dataset = AudioVideoDataset(
                    data_directory=self.root_data_dir,
                    clip_length=self.clip_length,
                    dataset_name=self.VISUAL_DATASET[1],
                    dataset_type="val",
                    data_split=self.split,
                    use_sound=self.use_sound,
                    add_noise=self.add_noise,
                )
        return self._val_dataset

    @property
    def test_dataset(self):
        if (self._test_dataset == None) and (self.dataset_name in self.VISUAL_DATASET):
            if self.dataset_name == self.VISUAL_DATASET[0]:
                self._test_dataset = DHF1KDataset(
                    data_directory=os.path.join(self.root_data_dir, "test"),
                    clip_length=self.clip_length,
                    multi_frame=self.multi_frame,
                    dataset_type="test",
                    clip_step=self.clip_step,
                )
        return self._test_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloaders(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
