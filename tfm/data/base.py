import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from tfm.constants import LIGHTS_DATASET
from tfm.data.lights import LightsOutDataset
from tfm.data.puzzle import Puzzle8MnistDataset


class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, input_size: int, num_workers: int, dataset: str):
        super().__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.dataset_size = self.input_size
        self.num_workers = num_workers

        self.dataset_class = Puzzle8MnistDataset
        if dataset == LIGHTS_DATASET:
            self.dataset_size = 5
            self.dataset_class = LightsOutDataset

        self.training = None
        self.evaluation = None

    def setup(self, stage: str):
        transformations = torch.nn.Sequential(
            transforms.Resize((self.input_size, self.input_size)),
            transforms.RandomResizedCrop((self.input_size, self.input_size), scale=(0.95, 1.0)),
            transforms.RandomRotation(15),
            transforms.GaussianBlur(3),
        )

        self.training = self.dataset_class(self.dataset_size, 100, self.batch_size, transformations=transformations)
        self.evaluation = self.dataset_class(self.dataset_size, 16, self.batch_size)

    def train_dataloader(self):
        return DataLoader(self.training, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.evaluation, batch_size=self.batch_size, num_workers=self.num_workers)
