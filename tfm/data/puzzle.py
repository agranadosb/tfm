import random
from typing import List, Tuple

import numpy as np
import torch
import torchvision
from torch.nn.functional import one_hot
from torchvision import transforms

from tfm.constants import ORDERED_ORDER, MOVEMENTS, MOVEMENT_TO_LABEL
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
        self.size = 28 * 3
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

    def random_permutation(self, current_order: np.ndarray, zero_index: int) -> Tuple[int, int]:
        """Returns a new order with one random movement from a current order.

        Parameters
        ----------
        current_order: np.ndarray
            Current order of the puzzle. This parameter will be modified, to
            prevent it copy the parameter before calling the function.
        zero_index:
            Index where the zero is at.

        Returns
        -------
        movement: int
            Movement selected, could be one of:
                 - 3 -> Up
                 - -3 -> Bottom
                 - 1 -> Right
                 - -1 -> Left
        new_index: int
            New index of the zero on the order.
        """
        random_movement = random.choice(MOVEMENTS)
        new_index = zero_index + random_movement

        beyond_bounds = new_index < 0 or new_index > 8
        incorrect_left = zero_index % 3 == 0 and random_movement == -1
        incorrect_right = zero_index % 3 == 2 and random_movement == 1

        if beyond_bounds or incorrect_left or incorrect_right:
            random_movement = -1 * random_movement
            new_index = zero_index + random_movement

        current_order[[zero_index, new_index]] = current_order[new_index], 0

        return random_movement, new_index

    def get_image(self, sequence: np.ndarray) -> torch.Tensor:
        digits_selection = np.zeros(len(sequence), dtype=np.int16)
        for index, digit in enumerate(sequence):
            digits_selection[index] = np.random.choice(self.indices[digit])

        return self._get(digits_selection)[0]

    def get_batch(self, batch_size: int, to_label: bool = True, to_one_hot: bool = True, shuffle: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data = torch.zeros((batch_size * 4, 1, self.size, self.size), dtype=torch.float)
        movements = torch.zeros(batch_size * 4, dtype=torch.int64)
        transitions = torch.zeros((batch_size * 4, 1, self.size, self.size), dtype=torch.float)
        current_order = self.order.copy()
        zero_index = 4

        for i in range(batch_size * 4):
            previous_order = current_order.copy()
            source = self.get_image(previous_order)
            data[i] = torch.unsqueeze(source, 0)

            movement, zero_index = self.random_permutation(current_order, zero_index)

            transition = self.get_image(current_order)

            transitions[i] = torch.unsqueeze(transition, 0)
            if to_label:
                movement = MOVEMENT_TO_LABEL[movement]
            movements[i] = movement

        if to_one_hot:
            movements = one_hot(movements, num_classes=4)

        permutation = torch.randperm(batch_size * 4)

        data = data[permutation][:batch_size]
        transitions = transitions[permutation][:batch_size]
        movements = movements[permutation][:batch_size]

        data.requires_grad = True

        return data, transitions, movements

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
