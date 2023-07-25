import sys

import fire
import torch
from fire.core import FireExit

from tfm.utils.cli import get_cli

torch.set_float32_matmul_precision("medium")


if __name__ == "__main__":
    try:
        fire.Fire(get_cli())
    except FireExit:
        sys.exit(1)
