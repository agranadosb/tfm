import os
import fire
import torch

from tfm.cli.inference import random_prediction, move, show_possibilities
from tfm.cli.training import train, save_checkpoint
from tfm.cli.tuning import hyperparameter_tuning

os.environ["WANDB_API_KEY"] = "local-c0e251ebaef9040fd778217f91fcff426f891404"

torch.set_float32_matmul_precision('medium')


if __name__ == "__main__":
    fire.Fire({
        "train": train,
        "hyperparameter-tuning": hyperparameter_tuning,
        "save-checkpoint": save_checkpoint,
        "random-prediction": random_prediction,
        "move": move,
        "show-possibilities": show_possibilities,
    })
