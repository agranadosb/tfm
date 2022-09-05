import random
from typing import List, Tuple, Optional

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

    def _get(self, indices: List[int]) -> Tuple[torch.Tensor, List[int]]:
        """Returns the 8-puzzle wrote on 'sequence'.

        Parameters
        ----------
        indices: List[int]
            Indices of the digits on the dataset.

        Returns
        -------
        Tuple[np.ndarray, List[int]]
            8-Puzzle generated.
            Order of the digits on the mnist puzzle."""
        digits = [-1 for _ in range(9)]
        for column in range(3):
            for row in range(3):
                ymin = column * 28
                xmin = row * 28
                ymax = ymin + 28
                xmax = xmin + 28
                index = row * 3 + column
                image, digit = self.dataset[indices[index]]
                digits[index] = digit
                self.base_image[xmin:xmax, ymin:ymax] = image

        return self.base_image, digits

    def get(
        self, ordered: bool = False, sequence: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor, List[int]]:
        """Returns a random generated 8-puzzle.

        Parameters
        ----------
        ordered: bool = False
            If True the 8-puzzle will be ordered.
        sequence: Optional[List[int]]
            If given this sequence is used instead of random selected digits.

        Returns
        -------
        Tuple[np.ndarray, List[int]]
            8-Puzzle generated.
            Order of the digits on the mnist puzzle."""
        empty_sequence = sequence is None
        if empty_sequence:
            sequence = range(9)
        digits_selection = [random.choice(self.indices[digit]) for digit in sequence]

        if not ordered and empty_sequence:
            random.shuffle(digits_selection)

        return self._get(digits_selection)
