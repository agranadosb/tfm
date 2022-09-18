from typing import List, Tuple, Optional, Union

import numpy as np
import torch
import torchvision
from torchvision import transforms

from tfm.constants import ORDERED_ORDER, MOVEMENTS
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
        self.order = order

        if not has_correct_order(order):
            raise ValueError("The list must have only the next values")

        self.indices = {
            index: np.zeros(different_digits, dtype=np.int) for index in range(10)
        }
        index_number = {index: -1 for index in range(10)}
        completed = np.zeros(10, dtype=np.bool)
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

    def _random_movements(
        self, total_movements: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        current_index = self.order.index(0)
        current_order = np.asarray(self.order)
        choice = np.random.choice

        movements = np.zeros(total_movements)
        available_movements = np.asarray(MOVEMENTS)
        selected_movements = choice(available_movements, total_movements)
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

    def get(
        self,
        ordered: bool = False,
        sequence: Optional[Union[List[int], np.ndarray]] = None,
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

        if not ordered and empty_sequence:
            sequence = self._random_movements()[0]

        digits_selection = [np.random.choice(self.indices[digit]) for digit in sequence]

        return self._get(digits_selection)
