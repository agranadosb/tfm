import datetime
import os
from typing import Dict, Any

import torch
from pytorch_lightning.loggers import WandbLogger
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
import pytorch_lightning as pl

from tfm.data.puzzle import Puzzle8MnistDataModule
from tfm.model.base import ConvBlock
from tfm.model.trainer import Trainer

os.environ["WANDB_API_KEY"] = "local-c0e251ebaef9040fd778217f91fcff426f891404"

torch.set_float32_matmul_precision('medium')


def train_model(config: Dict[str, Any], project, epochs: int = 10):
    metrics = {"loss": "ptl/val_loss", "acc": "ptl/val_accuracy"}
    callbacks = [TuneReportCallback(metrics, on="validation_end")]

    trainer = pl.Trainer(
        max_epochs=epochs,
        logger=WandbLogger(
            project=project,
            name=f"trial-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
        ),
        callbacks=callbacks,
    )
    trainer.fit(
        model=Trainer(config),
        datamodule=Puzzle8MnistDataModule(
            config["batch_size"],
            config["input_size"],
            config["num_workers"]
        )
    )


if __name__ == "__main__":
    # TODO: Export the configuration into a file and load it
    # TODO: Use argparse to select the function to execute (one main argparse and one for each function)
    configuration = {
        # Dataset configuration
        "input_size": 96,
        "batch_size": tune.choice([8, 16, 32, 64, 128]),
        "num_workers": 4,
        # Training configuration
        "lr": tune.loguniform(1e-4, 1e-2),
        "weight_decay": tune.loguniform(1e-3, 1e-1),
        "num_movement": 4,
        # Start model configuration
        "layers": tune.choice([
            [8, 16, 32, 64, 128],
            [8, 32, 128],
            [16, 32, 64, 128, 256],
            [16, 64, 256],
            [32, 64, 128],
            [16, 32, 64, 128],
            [16, 32, 64, 128, 256],
        ]),
        "movements_layers": tune.choice([
            [256, 64, 16],
            [512, 256, 128, 64],
            [256, 128, 64, 32],
            [128, 64, 32, 16, 8],
            [64, 32, 16, 8, 4],
        ]),
        "out_channels": 1,
        "input_channels": 1,
        "movements": 4,
        "block": ConvBlock,
    }

    analysis = tune.run(
        tune.with_parameters(
            train_model,
            epochs=10,
            project=(
                f"hyperparameter-tuning-"
                f"{ConvBlock.__name__}-"
                f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )
        ),
        config=configuration,
        resources_per_trial={
            "gpu": 1
        },
        num_samples=30
    )
