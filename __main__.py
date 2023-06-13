import os
import fire
import torch
from tfm.cli.training import train
from tfm.cli.tuning import hyperparameter_tuning

os.environ["WANDB_API_KEY"] = "local-c0e251ebaef9040fd778217f91fcff426f891404"

torch.set_float32_matmul_precision('medium')


if __name__ == "__main__":
    fire.Fire(
        {"training": train},
        {"hyperparameter-tuning": hyperparameter_tuning}
    )
