from typing import Tuple

import torch
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from torchvision.transforms import transforms

from tfm.constants import LABEL_TO_MOVEMENT, LABEL_TO_STRING
from tfm.data.puzzle import Puzzle8MnistDataset, Puzzle8MnistGenerator
from tfm.plots.images import plot_image, plot_sample_result


def random_prediction(model_path: str):
    with torch.inference_mode():
        model = torch.load(model_path)
        image = Puzzle8MnistDataset(1, 1, 96)[0][0].unsqueeze(0).to(model.device)
        plot_image(model.model(image)[0].squeeze().cpu().detach())


def show_possibilities():
    indices = [0, 1, 3, 4]
    generator = Puzzle8MnistDataset(1, 1, 96)
    initial_image, images_moved, movements = generator[0]

    fig = plt.figure(constrained_layout=True)
    gs = GridSpec(2, 5, figure=fig)

    fig.add_subplot(gs[0, :])

    plt.imshow(initial_image.squeeze())
    plt.axis('off')
    plt.title("Input Image", fontsize=40)

    for index, image, movement in zip(indices, images_moved, movements):
        fig.add_subplot(gs[1, index])
        plt.imshow(image.squeeze())
        plt.axis('off')
        movement_index = movement.argmax().item()
        movement_str = "Not Possible"
        if movement_index < 4:
            movement_str = LABEL_TO_STRING[movement_index]

        movement_list_str = (
            str(movement.numpy().tolist())
            .replace("[", "")
            .replace("]", "")
            .replace(",", " ")
        )
        plt.title(f"{movement_list_str} - {movement_str}", fontsize=20)

    plt.show()


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
