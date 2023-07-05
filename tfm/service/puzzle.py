import logging
from typing import Tuple

import torch
from bentoml import Runner
from torchvision.transforms import transforms
from torchvision.utils import make_grid

from tfm.constants import LABEL_TO_STRING
from tfm.data.puzzle import Puzzle8MnistGenerator


class Puzzle8MnistService:
    logger = logging.getLogger("bentoml")

    def __init__(self, model: Runner):
        self.model = model

        self.resize = transforms.Resize((96, 96))
        self.to_image = transforms.ToPILImage()

    def predict(self, order: Tuple[int, int, int, int, int, int, int, int, int]):
        image = Puzzle8MnistGenerator().get(order).unsqueeze(0).unsqueeze(0)
        image = self.resize(image)
        result = self.model.run(image)

        result_image = result[0].squeeze(0).detach().to("cpu")
        movement = LABEL_TO_STRING["puzzle8"][
            result[1].squeeze().argmax().detach().to("cpu").item()
        ]

        self.logger.info(f"Movement: {movement}")

        grid_images = torch.zeros((2, 1, 96, 96))
        grid_images[0] = image.squeeze(0)
        grid_images[1] = result_image

        return self.to_image(
            make_grid(grid_images, nrow=2, padding=5, pad_value=1)
        )
