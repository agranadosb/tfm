from typing import Tuple

import torch
from torchvision.transforms import transforms

from tfm.constants import LABEL_TO_MOVEMENT
from tfm.data.puzzle import Puzzle8MnistDataset, Puzzle8MnistGenerator
from tfm.plots.images import plot_image, plot_sample_result


def random_prediction(model_path: str):
    with torch.inference_mode():
        model = torch.load(model_path)
        image = Puzzle8MnistDataset(1, 1, 96)[0][0].unsqueeze(0).to(model.device)
        plot_image(model.model(image)[0].squeeze().cpu().detach())


def move(order: Tuple[int], model_path: str):
    order = torch.tensor(list(order))
    image = transforms.Resize((96, 96))(
        Puzzle8MnistGenerator().get(order).unsqueeze(0)
    ).unsqueeze(0)
    with torch.inference_mode():
        model = torch.load(model_path)
        image = image.to(model.device)
        result = model.model(image)

        image = image.squeeze().cpu().detach().squeeze()
        result_image = result[0].cpu().detach().squeeze()
        threshold = 0.42
        result_image[result_image < threshold] = 0
        result_movement = LABEL_TO_MOVEMENT[
            result[1].cpu().detach().argmax(dim=-1).squeeze().item()
        ]

        plot_sample_result(image, result_image, result_movement)
