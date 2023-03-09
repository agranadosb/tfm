from typing import List, Tuple, Optional
from matplotlib import pyplot as plt
from PIL.Image import Image


def plot_images(images: List[Image], size: Tuple[int, int] = (3, 3), titles: Optional[List[str]] = None):
    """Plots a grid of images.

    Parameters
    ----------
    images: List[Image]
        List of images to plot on the grid.
    size: Tuple[int, int] = (3, 3)
        Size of the grid.
    titles: Optional[str]
        Titles to give to the plots."""
    if titles is None:
        titles = [""] * len(images)
    _, grid = plt.subplots(*size, figsize=(12, 12))
    grid = grid.flatten()
    for title, img, ax in zip(titles, images, grid):
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.title.set_text(title)
        ax.imshow(img)
    plt.show()
