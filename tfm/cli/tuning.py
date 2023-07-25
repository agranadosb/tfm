from typing import Dict, Any

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger  # noqa
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch

from tfm.constants import BLOCKS, PUZZLE_DATASET
from tfm.data.base import DataModule
from tfm.model.trainer import Trainer
from tfm.utils.data import current_datetime


def _train_model(config: Dict[str, Any], project, epochs: int = 10):
    logger = None
    try:
        logger = WandbLogger(project=project, name=f"trial-{current_datetime()}")
    except ModuleNotFoundError:
        print("Wandb not found. Logging won't be done.")
        config["log"] = False

    trainer = pl.Trainer(max_epochs=epochs, logger=logger)

    trainer.fit(
        model=Trainer(config),
        datamodule=DataModule(
            config["batch_size"],
            config["input_size"],
            config["num_workers"],
            PUZZLE_DATASET,
        ),
    )


def hyperparameter_tuning(block: str, samples: int, epochs: int):
    """
    Create a hyperparameter tuning experiment using Ray Tune. The experiment
    will use the block passed as parameter and will run the experiment
    using the number of samples and epochs passed as parameters.

    Parameters
    ----------
    block: str
        The block to use in the experiment
    samples: int
        The number of samples to use in the experiment
    epochs:
        The number of epochs to use in the experiment
    """
    configuration = {
        # Dataset configuration
        "input_size": 96,
        "batch_size": 128,
        "num_workers": 4,
        "num_movement": 4,
        "out_channels": 1,
        "input_channels": 1,
        "movements": 4,
        "block": BLOCKS[block],
        "hyp": True,
        "dataset": PUZZLE_DATASET
    }
    search_space = {
        # Training configuration
        "lr": tune.loguniform(1e-4, 1e-3),
        "weight_decay": tune.loguniform(1e-7, 1e-6),
        # Start model configuration
        "layers": tune.choice(
            [
                [4, 16, 64],
                [4, 8, 16, 32],
                [8, 32, 128],
                [8, 16, 32, 64],
            ]
        ),
        "movements_layers": tune.choice(
            [
                [64],
                [32],
                [16],
            ]
        ),
    }

    hyperopt_search = HyperOptSearch(search_space, metric="val_loss", mode="min")

    results = tune.run(
        tune.with_parameters(
            _train_model, epochs=epochs, project=f"hpt-{block}-{current_datetime()}"
        ),
        config=configuration,
        search_alg=hyperopt_search,
        resources_per_trial={"gpu": 1},
        num_samples=samples,
    )

    best_result = results.get_best_result(metric="ptl/val_accuracy", mode="max")
    print(best_result.log_dir)
