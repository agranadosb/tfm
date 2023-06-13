import os
from datetime import datetime
from typing import Dict, Any, Union

import torch
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl

from tfm.constants import BLOCKS
from tfm.data.puzzle import Puzzle8MnistDataModule
from tfm.model.trainer import Trainer


def train(config: Union[Dict[str, Any], str], project, epochs: int):
    """
    This function trains and evaluates a model using the configuration provided.

    A configuration file is a dictionary with the following structure:

    ```yaml
    # Dataset configuration
    input_size: 96
    batch_size: 64
    num_workers: 4
    # Training configuration
    lr: 0.0009299177250712636
    weight_decay: 0.09734765783141108
    num_movement: 4
    # Start model configuration
    layers: [16, 32, 64, 128, 256]
    movements_layers: [128, 64, 32, 16, 8]
    out_channels: 1
    input_channels: 1
    movements: 4
    block: conv
    ```

    Parameters
    ----------
    config: Union[Dict[str, Any], str]
        Yaml configuration file or dictionary with the configuration.
    project: str
        Name of the project that will be used in wandb.
    epochs: int
        Number of epochs to train the model.
    """
    if isinstance(config, str):
        with open(config, 'r') as stream:
            config = yaml.safe_load(stream)

            config["block"] = BLOCKS[config["block"]]

    project = f"{project}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(os.getcwd(), project),
        save_top_k=3,
        verbose=True,
        monitor='ptl/val_accuracy',
        mode='max'
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        logger=WandbLogger(
            project=project,
            name="full-training",
        ),
        enable_checkpointing=True,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(
        model=Trainer(config),
        datamodule=Puzzle8MnistDataModule(
            config["batch_size"],
            config["input_size"],
            config["num_workers"]
        )
    )


def save_checkpoint(path: str, path_to_save: str, config: str):
    if isinstance(config, str):
        with open(config, 'r') as stream:
            config = yaml.safe_load(stream)

            config["block"] = BLOCKS[config["block"]]
    model = Trainer.load_from_checkpoint(path, config=config)
    torch.save(model, path_to_save)
