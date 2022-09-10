import random
from typing import List, Tuple, Optional

import numpy as np
import torch
import torchvision
from torchvision import transforms

from tfm.constants import ORDERED_ORDER
from tfm.utils.puzzle import has_correct_order


class Puzzle8MnistGenerator:
    """This class generates a random 8.puzzle based on mnist digits.

    Parameters
    ----------
    train: bool = True
        Indicates if the split of the dataset if for training or not.
    order: List[int] = tuple(range(9))
        Default order on the grid. The default order is:
        ```
        0 1 2
        3 4 5
        6 7 8
        ```"""

    def __init__(self, train: bool = True, order: List[int] = ORDERED_ORDER):
        self.train = train
        self.transformation = transforms.ToTensor()
        self.dataset = torchvision.datasets.MNIST(
            root="./data", train=train, download=True
        )
        self.order = order

        if not has_correct_order(order):
            raise ValueError("The list must have only the next values")

        self.indices = {index: [] for index in range(10)}
        for index in range(len(self.dataset)):
            _, digit = self.dataset[index]
            self.indices[digit].append(index)

        # Indices to numpy for fast access to indices
        for digit, digit_indices in self.indices.items():
            self.indices[digit] = np.asarray(digit_indices)

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
                image = self.transformation(image)

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
            sequence = self.order
        digits_selection = [np.random.choice(self.indices[digit]) for digit in sequence]

        if not ordered and empty_sequence:
            random.shuffle(digits_selection)

        return self._get(digits_selection)
