import os
from typing import Any

from lightning import Trainer as LightningTrainer
from lightning.pytorch.loggers import WandbLogger
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch

from tfm.conf.schema import Configuration
from tfm.data.dataset import BaseDataModule
from tfm.model.trainer import Trainer
from tfm.utils.data import current_datetime


def _train_model(config: dict[str, Any], project, epochs: int = 10):
    os.environ["WANDB_API_KEY"] = "local-2cdcb89d8ebf517dcfc2ac3c36e14a740d235c15"

    dataset = {
        "input_size": config["input_size"],
        "batch_size": config["batch_size"],
        "num_workers": config["num_workers"],
        "dataset": config["dataset"]
    }
    training = {
        "lr": config["lr"],
        "weight_decay": config["weight_decay"],
        "training": 0.6,
        "validation": 0.2,
        "hyp": True
    }
    model = {
        "layers": config["layers"],
        "movements_layers": config["movements_layers"],
        "out_channels": config["out_channels"],
        "input_channels": config["input_channels"],
        "actions": config["actions"],
        "block": config["block"],
        "log": True
    }

    configuration = Configuration(**{"dataset": dataset, "training": training, "model": model})

    logger = None
    run_name = f"full-training-{current_datetime()}"
    if configuration.model.log:
        try:
            logger = WandbLogger(project=project, name=run_name)
        except ModuleNotFoundError:
            print("Wandb not found. Logging won't be done.")
            configuration.model.log = False

    trainer = LightningTrainer(
        max_epochs=epochs,
        logger=logger,
    )

    trainer.fit(
        model=Trainer(configuration),
        datamodule=BaseDataModule("/opt/projects/tfm/data", configuration),
    )


def hyperparameter_tuning(
    block: str,
    samples: int,
    epochs: int,
    dataset: str,
    actions: int
):
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
        "input_size": 96,
        "batch_size": 32,
        "num_workers": 4,
        "actions": actions,
        "out_channels": 1,
        "input_channels": 1,
        "block": block,
        "dataset": dataset,
        "layers": [4, 16, 64],
        "movements_layers": [64, 32],
    }
    search_space = {
        # Training configuration
        "lr": tune.loguniform(1e-4, 1e-2),
        "weight_decay": tune.loguniform(1e-7, 1e-3),
    }

    hyperopt_search = HyperOptSearch(search_space, metric="val_loss", mode="min")

    results = tune.run(
        tune.with_parameters(
            _train_model, epochs=epochs, project=f"hpt-{block}-{current_datetime()}"
        ),
        config=configuration,
        search_alg=hyperopt_search,
        resources_per_trial={"gpu": 1, "cpu": 8},
        num_samples=samples,
    )

    best_result = results.get_best_result(metric="val_loss", mode="min")
    print(best_result.log_dir)
