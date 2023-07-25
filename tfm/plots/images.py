import torch
from matplotlib import pyplot as plt


def plot_image(image: torch.Tensor):
    """
    Plots a torch tensor as an image.

    Parameters
    ----------
    image

    Returns
    -------

    """
    if image.ndim == 2:
        image = image.squeeze(-1)
    elif image.ndim == 3:
        image = image.permute(1, 2, 0)

    plt.axis("off")
    plt.imshow(image)
    plt.show()


def plot_sample_result(
    input_image: torch.Tensor, predicted_image: torch.Tensor, predicted_label: int
):
    """
    Plots a sample, its prediction and the predicted label.

    Parameters
    ----------
    input_image: torch.Tensor
        The input image.
    predicted_image: torch.Tensor
        The predicted image.
    predicted_label: int
        The predicted label.

    Returns
    -------

    """

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(input_image)
    axs[0].set_title("Input Image")
    axs[0].axis("off")
    fig.suptitle(f"Predicted movement: {predicted_label}", fontsize=16)

    axs[1].imshow(predicted_image)
    axs[1].set_title("Predicted Movement Image")
    axs[1].axis("off")

    plt.show()
