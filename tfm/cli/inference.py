import os
from typing import Optional

import bentoml
import torch
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

from tfm.constants import LABEL_TO_STRING, LIGHTS_DATASET
from tfm.data.lights import LightsOutDataset
from tfm.data.puzzle import Puzzle8MnistDataset
from tfm.utils.data import current_datetime


def show_possibilities_lights():
    """This class shows all the possible movements for a 3x3 lights out puzzle."""

    n = 3
    indices = [i for i in range(n * n)]
    generator = LightsOutDataset(3, 1, 1)
    initial_image, images_moved, movements = generator[0]

    fig = plt.figure(constrained_layout=True)
    plt.gray()
    gs = GridSpec(4, 3, figure=fig)

    fig.add_subplot(gs[0, :])

    plt.imshow(initial_image.squeeze())
    plt.axis("off")
    plt.title("Input Image", fontsize=40)

    for index, image, movement in zip(indices, images_moved, movements):
        row = index // n + 1
        col = index % n
        fig.add_subplot(gs[row, col])
        plt.imshow(image.squeeze())
        plt.axis("off")
        movement_str = LABEL_TO_STRING["lights-out"][index]

        movement_list_str = (
            str(movement.numpy().tolist())
            .replace("[", "")
            .replace("]", "")
            .replace(",", " ")
        )
        plt.title(f"{movement_list_str} - {movement_str}", fontsize=20)

    plt.show()


def show_possibilities_puzzle():
    """This class shows all the possible movements for an 8-puzzle."""

    indices = [0, 1, 3, 4]
    generator = Puzzle8MnistDataset(96, 1, 1)
    initial_image, images_moved, movements = generator[0]

    fig = plt.figure(constrained_layout=True)
    plt.gray()
    gs = GridSpec(2, 5, figure=fig)

    fig.add_subplot(gs[0, :])

    plt.imshow(initial_image.squeeze())
    plt.axis("off")
    plt.title("Input Image", fontsize=40)

    for index, image, movement in zip(indices, images_moved, movements):
        fig.add_subplot(gs[1, index])
        plt.imshow(image.squeeze())
        plt.axis("off")
        movement_index = movement.argmax().item()
        movement_str = "Not Possible"
        if movement_index < 4:
            movement_str = LABEL_TO_STRING["puzzle8"][movement_index]

        movement_list_str = (
            str(movement.numpy().tolist())
            .replace("[", "")
            .replace("]", "")
            .replace(",", " ")
        )
        plt.title(f"{movement_list_str} - {movement_str}", fontsize=20)

    plt.show()


def generate_samples(
    dataset_str: str, model_name: str, n: int, base_path: Optional[str] = None
):
    """
    This function generates n samples from the dataset, then those samples are
    passed through the model. Finally, the input and output images are saved in
    the pwd folder. The model used has to be saved by bentoml and the model
    name has to be `model_name` argument. You can provide a base path to save
    the results in a different folder.


    Parameters
    ----------
    dataset_str
        Name of the dataset to use. It can be `lights-out` or `puzzle8`.
    model_name
        Name of the model to use. It has to be saved by bentoml.
    n
        Number of samples to generate.
    base_path
        Path to save the results. If None, the current working directory is used.
    """
    dataset_class = Puzzle8MnistDataset
    dataset_params = {"num_batches": 1, "batch_size": n, "size": 96}
    if dataset_str == LIGHTS_DATASET:
        dataset_class = LightsOutDataset
        dataset_params["size"] = 5

    dataset = DataLoader(dataset_class(**dataset_params), batch_size=n)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.inference_mode():
        model = bentoml.pytorch.load_model(model_name).to(device)
        for images, _, _ in dataset:
            images = images.to(device)
            result = model(images)

    if base_path is None:
        base_path = os.getcwd()

    if not os.path.exists(base_path):
        os.mkdir(base_path)

    results_folder = os.path.join(
        base_path, f"{dataset_str}-results-{current_datetime()}"
    )
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)

    with open(f"{results_folder}/{dataset_str}_samples.txt", "w") as f:
        for index, (image, image_moved, movement) in enumerate(
            zip(images, result[0], result[1])
        ):
            image = to_pil_image(image.cpu().detach())
            image_moved = to_pil_image(image_moved.cpu().detach())
            movement = movement.cpu().detach().argmax(dim=-1).squeeze().item()
            movement_str = LABEL_TO_STRING[dataset_str][movement]

            f.write(f"{index}-{movement_str}\n")
            image.save(f"{results_folder}/{index}_input.png")
            image_moved.save(f"{results_folder}/{index}_output.png")
