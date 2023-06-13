import torch

from tfm.data.puzzle import Puzzle8MnistDataset
from tfm.plots.images import plot_image


def random_prediction(model_path: str):
    with torch.inference_mode():
        model = torch.load(model_path)
        image = Puzzle8MnistDataset(1, 1, 96)[0][0].unsqueeze(0).to(model.device)
        plot_image(model.model(image)[0].squeeze().cpu().detach())
