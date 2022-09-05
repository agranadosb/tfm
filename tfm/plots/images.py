from typing import List, Tuple
from matplotlib import pyplot as plt
from PIL.Image import Image


def plot_images(
    images: List[Image], size: Tuple[int, int] = (3, 3)
):
    """Plots a grid of images.

    Parameters
    ----------
    images: List[Image]
        List of images to plot on the grid.
    size: Tuple[int, int] = (3, 3)
        Size of the grid."""
    _, grid = plt.subplots(*size, figsize=(12, 12))
    grid = grid.flatten()
    for img, ax in zip(images, grid):
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.imshow(img)
    plt.show()
