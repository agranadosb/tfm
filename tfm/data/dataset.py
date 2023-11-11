from pathlib import Path

import torch
import torchvision
from lightning import LightningDataModule
from torch import Tensor
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode

from tfm.conf.schema import Configuration


class BaseDataset(Dataset):
    def __init__(
        self,
        path: str | list[Path],
        actions: int,
        resize: Module,
        transform: Module = None,
    ):
        self.actions = actions
        self.resize = resize
        self.transform = transform

        self.samples = path
        if isinstance(path, str):
            self.samples = list(Path(path).iterdir())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item) -> tuple[Tensor, Tensor, Tensor]:
        sample_path = self.samples[item]

        state_path = sample_path / "state" / "state.png"
        actions_path = list((sample_path / "actions").iterdir())
        actions_indices = [int(path.stem) for path in actions_path]

        state = self.resize(read_image(str(state_path), mode=ImageReadMode.GRAY))
        actions = [
            self.resize(read_image(str(path), mode=ImageReadMode.GRAY)).unsqueeze(0)
            for path in actions_path
        ]

        if self.transform is not None:
            state = self.transform(state)

        sample_actions_ids = torch.tensor(actions_indices, dtype=torch.int64)
        sample_actions_images = torch.cat(actions)

        data_actions_images = torch.zeros(
            (self.actions, *sample_actions_images.shape[1:]),
            dtype=sample_actions_images.dtype,
        )
        data_actions_images[sample_actions_ids] = sample_actions_images

        data_actions_ids = torch.zeros(
            (self.actions, self.actions + 1), dtype=torch.int64
        )
        data_actions_ids[..., -1] = 1
        data_actions_ids[sample_actions_ids, -1] = 0
        data_actions_ids[torch.arange(self.actions), torch.arange(self.actions)] = 1

        return state / 255.0, data_actions_images.squeeze(1) / 255.0, data_actions_ids

    def loader(
        self, batch_size: int = 1, shuffle: bool = True, workers: int = 0
    ) -> DataLoader:
        return DataLoader(
            self, batch_size=batch_size, shuffle=shuffle, num_workers=workers
        )


class BaseDataModule(LightningDataModule):
    train_dataset: BaseDataset
    val_dataset: BaseDataset
    test_dataset: BaseDataset

    def __init__(self, path: str, config: Configuration):
        super().__init__()
        self.path = path
        self.config = config

        self.samples = list(Path(self.path).iterdir())

        self.training = config.training.training
        self.validation = config.training.validation

        self.train_samples = int(len(self.samples) * config.training.training)
        self.val_samples = int(len(self.samples) * config.training.validation)
        self.test_samples = len(self.samples) - self.train_samples - self.val_samples

        self.resize = torchvision.transforms.Resize(
            (config.dataset.input_size, config.dataset.input_size)
        )

    def setup(self, stage: str):
        self.train_dataset = BaseDataset(
            self.samples[: self.train_samples], self.config.model.actions, self.resize
        )
        self.val_dataset = BaseDataset(
            self.samples[self.train_samples : self.train_samples + self.val_samples],
            self.config.model.actions,
            self.resize,
        )
        self.test_dataset = BaseDataset(
            self.samples[self.train_samples + self.val_samples :],
            self.config.model.actions,
            self.resize,
        )

    def train_dataloader(self):
        return self.train_dataset.loader(
            self.config.dataset.batch_size, workers=self.config.dataset.num_workers
        )

    def val_dataloader(self):
        return self.val_dataset.loader(
            self.config.dataset.batch_size, workers=self.config.dataset.num_workers
        )

    def test_dataloader(self):
        return self.test_dataset.loader(
            self.config.dataset.batch_size, workers=self.config.dataset.num_workers
        )
