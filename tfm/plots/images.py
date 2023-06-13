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

    plt.imshow(image)
    plt.show()
