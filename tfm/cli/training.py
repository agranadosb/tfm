import os
from typing import Dict, Any, Union

import bentoml
import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger  # noqa

from tfm.constants import BLOCKS
from tfm.data.base import DataModule
from tfm.model.trainer import Trainer
from tfm.utils.data import current_datetime


def train(config: Union[Dict[str, Any], str], project: str, epochs: int):
    """
    This function trains and evaluates a model using the configuration provided.

    A configuration file is a dictionary with the following structure:

    ```yaml
    # Dataset configuration
    input_size: 96
    batch_size: 64
    num_workers: 4
    # Training configuration
    lr: 0.001
    weight_decay: 0.1
    num_movement: 4
    # Start model configuration
    layers: [16, 32, 64, 128, 256]
    movements_layers: [128, 64, 32, 16, 8]
    out_channels: 1
    input_channels: 1
    movements: 4
    block: conv
    log: False
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
        with open(config, "r") as stream:
            config = yaml.safe_load(stream)

            config["block"] = BLOCKS[config["block"]]

    logger = None
    run_name = f"full-training-{current_datetime()}"
    if config.get("log", True):
        try:
            logger = WandbLogger(project=project, name=run_name)
        except ModuleNotFoundError:
            print("Wandb not found. Logging won't be done.")
            config["log"] = False

    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(os.getcwd(), project, run_name),
            save_top_k=3,
            verbose=True,
            monitor="ptl/val_loss",
            mode="min",
        )
    ]

    trainer = pl.Trainer(
        max_epochs=epochs,
        logger=logger,
        enable_checkpointing=True,
        callbacks=callbacks,
    )
    trainer.fit(
        model=Trainer(config),
        datamodule=DataModule(
            config["batch_size"],
            config["input_size"],
            config["num_workers"],
            config["dataset"],
        ),
    )


def save_model(checkpoint: str, config: str, model_name: str):
    """
    This function saves a model using the checkpoint provided. The checkpoint
    will be generated by the training of the model. The configuration file is
    the same as the one used in the training. The model name is the name that
    will be used to save the model in bentoml.

    Parameters
    ----------
    checkpoint: str
        Checkpoint generated by the training of the model.
    config
        Configuration file used in the training of the model.
    model_name
        Name that will be used to save the model in bentoml.
    """
    if isinstance(config, str):
        with open(config, "r") as stream:
            config = yaml.safe_load(stream)

            config["block"] = BLOCKS[config["block"]]

    trainer = Trainer.load_from_checkpoint(checkpoint, config=config)
    tag = bentoml.pytorch.save_model(
        model_name,
        trainer.model,
        signatures={"__call__": {"batchable": True, "batch_dim": 0}},
    )
    print(f"Saved model: {tag}")
