from typing import List, Tuple, Optional, Union

import numpy as np
import torch
import torchvision
from torchvision import transforms

from tfm.constants import ORDERED_ORDER, MOVEMENTS
from tfm.utils.data import to_numpy
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
        1 2 3
        8 0 4
        7 6 5
        ```
    different_digits: int = 10
        Diversity of the digits. If `different_digits` is set to 10 it means
        that each of the digits will have 10 different shapes."""

    def __init__(
        self,
        train: bool = True,
        order: List[int] = ORDERED_ORDER,
        different_digits: int = 10,
    ):
        self.train = train
        self.transformation = transforms.ToTensor()
        self.dataset = torchvision.datasets.MNIST(
            root="./data", train=train, download=True
        )
        self.order = to_numpy(order)

        if not has_correct_order(order):
            raise ValueError("The list must have only the next values")

        self.indices = {
            index: np.zeros(different_digits, dtype=np.int16) for index in range(10)
        }
        index_number = {index: -1 for index in range(10)}
        completed = np.zeros(10, dtype=bool)
        for index in range(len(self.dataset)):
            _, digit = self.dataset[index]

            completed[digit] = (
                completed[digit] or index_number[digit] == different_digits - 1
            )
            current_digit_list_index = (index_number[digit] + 1) % different_digits

            index_number[digit] = current_digit_list_index
            self.indices[digit][current_digit_list_index] = index

            if completed.all():
                break

        self.base_image = torch.zeros((28 * 3, 28 * 3))

    def _random_movements(
        self, total_movements: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a random list of movements and returns the list of
        movements with the result of the application of these movements to the
        ordered puzzle.

        Parameters
        ----------
        total_movements: int = 20
            Total of movements to perform.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Result of the application of these movements to the ordered puzzle.
            List of movements being:
                 - 3 -> Up
                 - -3 -> Bottom
                 - 1 -> Right
                 - -1 -> Left"""
        current_index = np.where(self.order == 0)[0][0]
        current_order = self.order.copy()

        movements = np.zeros(total_movements)
        available_movements = np.asarray(MOVEMENTS)
        selected_movements = np.random.choice(available_movements, total_movements)
        for i, movement in enumerate(selected_movements):
            new_index = current_index + movement

            beyond_bounds = new_index < 0 or new_index > 8
            incorrect_left = movement % 3 == 0 and movement == -1
            incorrect_right = movement % 3 == 2 and movement == 1

            if beyond_bounds or incorrect_left or incorrect_right:
                movement = -movement
                new_index = current_index + movement

            movements[i] = movement
            current_order[new_index], current_order[current_index] = (
                current_order[current_index],
                current_order[new_index],
            )
            current_index = new_index

        return current_order, movements

    def _get(self, indices: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
        """Returns the 8-puzzle wrote on 'sequence'.

        Parameters
        ----------
        indices: np.ndarray
            Indices of the digits on the dataset.

        Returns
        -------
        Tuple[torch.Tensor, np.ndarray]
            Image of the 8-Puzzle generated.
            Order of the digits on the mnist puzzle."""
        digits = np.zeros(9, dtype=np.int16)
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
        self,
        ordered: bool = False,
        sequence: Optional[Union[np.ndarray, List[int], Tuple[int, ...]]] = None,
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """Returns a random generated 8-puzzle.

        Parameters
        ----------
        ordered: bool = False
            If True the 8-puzzle will be ordered.
        sequence: Optional[Union[np.ndarray, List[int], Tuple[int, ...]]]
            If given this sequence is used instead of random selected digits.

        Returns
        -------
        Tuple[torch.Tensor, np.ndarray]
            Image of the 8-Puzzle generated.
            Order of the digits on the mnist puzzle."""
        empty_sequence = sequence is None
        if empty_sequence:
            sequence = self.order

        sequence = to_numpy(sequence)

        if not ordered and empty_sequence:
            sequence = self._random_movements()[0]

        digits_selection = np.zeros(len(sequence), dtype=np.int16)
        for index, digit in enumerate(sequence):
            digits_selection[index] = np.random.choice(self.indices[digit])

        return self._get(digits_selection)
