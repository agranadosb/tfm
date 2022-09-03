import random
from typing import List, Tuple

import numpy as np
import torch
import torchvision
from torchvision import transforms


class Puzzle8MnistGenerator:
    """This class generates a random 8.puzzle based on mnist digits.

    Parameters
    ----------
    train: bool = True
        Indicates if the split of the dataset if for training or not."""

    def __init__(self, train: bool = True):
        self.train = train
        self.dataset = torchvision.datasets.MNIST(
            root="./data", train=train, download=True, transform=transforms.ToTensor()
        )
        self.indices = {index: [] for index in range(10)}
        for index in range(len(self.dataset)):
            _, digit = self.dataset[index]
            self.indices[digit].append(index)
        del self.indices[9]
        self.base_image = torch.zeros((28 * 3, 28 * 3))

    def get(self, ordered: bool = False) -> Tuple[torch.Tensor, List[int]]:
        """Returns a random generated 8-puzzle.

        Parameters
        ----------
        ordered: bool = False
            If True the 8-puzzle will be ordered.

        Returns
        -------
        Tuple[np.ndarray, List[int]]
            8-Puzzle generated.
            Order of the digits on the mnist puzzle."""
        digits_selection = [random.choice(self.indices[digit]) for digit in range(9)]

        if not ordered:
            random.shuffle(digits_selection)

        digits = []
        for column in range(3):
            for row in range(3):
                ymin = column * 28
                xmin = row * 28
                ymax = ymin + 28
                xmax = xmin + 28
                image, digit = self.dataset[digits_selection[row * 3 + column]]
                self.base_image[xmin:xmax, ymin:ymax] = image
                digits.append(digit)

        return self.base_image, digits
