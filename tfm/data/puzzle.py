import random
from typing import List, Tuple, Sequence

import numpy as np
import torch
import torchvision
from torch import Tensor
from torch.nn.functional import one_hot
from torch.utils.data import Dataset
from torchvision import transforms

from tfm.constants import ORDERED_ORDER, MOVEMENTS, MOVEMENT_TO_LABEL, LABEL_TO_MOVEMENT
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
    """

    def __init__(
        self,
        train: bool = True,
        order: List[int] = ORDERED_ORDER,
    ):
        self.train = train
        self.size = 28 * 3
        self.dataset = torchvision.datasets.MNIST(
            root="./data", train=train, download=True, transform=transforms.ToTensor()
        )
        self.order = torch.IntTensor(order).to(torch.int8)

        if not has_correct_order(order):
            raise ValueError(
                "The list must have only the next values [0, 1, 2, 3, 4, 5, 6, 7, 8]"
            )

        self.indices = {index: np.zeros(10, dtype=np.int16) for index in range(10)}
        index_number = {index: -1 for index in range(10)}
        completed = np.zeros(10, dtype=bool)
        for index in range(len(self.dataset)):
            _, digit = self.dataset[index]

            completed[digit] = completed[digit] or index_number[digit] == 9
            current_digit_list_index = (index_number[digit] + 1) % 10

            index_number[digit] = current_digit_list_index
            self.indices[digit][current_digit_list_index] = index

            if completed.all():
                break

    def is_beyond_bounds(self, index: int) -> bool:  # noqa
        return index < 0 or index > 8

    def is_incorrect_left(self, index: int, movement: int) -> bool:  # noqa
        return index % 3 == 0 and movement == -1

    def is_incorrect_right(self, index: int, movement: int) -> bool:  # noqa
        return index % 3 == 2 and movement == 1

    def is_possible(self, zero_index: int, movement: int) -> bool:
        new_index = zero_index + movement

        return not (
            self.is_beyond_bounds(new_index)
            or self.is_incorrect_left(zero_index, movement)
            or self.is_incorrect_right(zero_index, movement)
        )

    def zero_index(self, order: Tensor) -> Union[int, np.int]:  # noqa
        return np.where(order == 0)[0][0]

    def move(self, order: Tensor, zero_index: int, movement: int) -> Tuple[Tensor, int]:
        new_index = zero_index + movement

        if not self.is_possible(zero_index, movement):
            random_movement = -1 * movement
            new_index = zero_index + random_movement

        current_order = order.clone()

        current_order[zero_index] = current_order[new_index]
        current_order[new_index] = 0

        return current_order, new_index

    def all_moves(
        self, order: Tensor, zero_index: int
    ) -> Tuple[Tensor, int, int, Tensor]:
        new_orders = torch.zeros((4, 9), dtype=torch.int8)
        applied_movements = torch.zeros(4, dtype=torch.int64)
        applied_movements += 4
        raw_movements = applied_movements.clone()

        for index, movement in enumerate(MOVEMENTS):
            if self.is_possible(zero_index, movement):
                new_order, _ = self.move(order, zero_index, movement)

                new_orders[index] = new_order
                applied_movements[index] = MOVEMENT_TO_LABEL[movement]
                raw_movements[index] = movement

        movement_to_apply = np.random.choice(raw_movements[raw_movements != 4])
        order, zero_index = self.move(order, zero_index, movement_to_apply)

        return (
            order,
            zero_index,
            MOVEMENT_TO_LABEL[movement_to_apply],
            applied_movements,
        )

    def random_move(
        self, current_order: Tensor, zero_index: int
    ) -> Tuple[Tensor, int, int]:
        movement = random.choice(MOVEMENTS)
        new_order, new_index = self.move(current_order, zero_index, movement)
        return new_order, new_index, MOVEMENT_TO_LABEL[movement]

    def random_sequence(
        self, size: int, all_moves: bool = False
    ) -> Tuple[Tensor, Tensor]:
        order = self.order.clone()
        zero_index = self.zero_index(order)

        total_size = size * 4
        moves_size = total_size
        if all_moves:
            moves_size = (total_size, 4)

        orders = torch.zeros((total_size, 9), dtype=torch.int8)
        movements = torch.zeros(moves_size, dtype=torch.int64)

        for i in range(total_size):
            method = self.random_move
            if all_moves:
                method = self.all_moves

            orders[i] = order
            order, zero_index, *_, movement = method(order, zero_index)
            movements[i] = movement

        choices = np.random.choice(np.arange(0, total_size), size)

        return orders[choices], movements[choices]

    def get(self, order: Sequence) -> torch.Tensor:
        """Returns the 8-puzzle wrote on 'sequence'.

        Parameters
        ----------
        order: np.ndarray
            Indices of the digits on the dataset.

        Returns
        -------
        torch.Tensor
            Image of the 8-Puzzle generated."""
        base_image = torch.zeros((28 * 3, 28 * 3))
        for column in range(3):
            for row in range(3):
                ymin = column * 28
                xmin = row * 28
                ymax = ymin + 28
                xmax = xmin + 28

                index = row * 3 + column
                digit = order[index].item()

                image, _ = self.dataset[np.random.choice(self.indices[digit])]

                base_image[xmin:xmax, ymin:ymax] = image

        return base_image


class Puzzle8MnistDataset(Dataset):
    def __init__(self, batches: int, batch_size: int):
        self.batch_size = batch_size
        self.batches = batches
        self._length = batches * batch_size

        self.data = Puzzle8MnistGenerator(order=[1, 2, 3, 8, 0, 4, 7, 6, 5])

        self.orders = torch.zeros((self._length, 9), dtype=torch.int8)
        self.movements = torch.zeros((self._length, 4), dtype=torch.int64)
        for i in range(batches):
            current_index = i * batch_size
            orders, moves = self.data.random_sequence(batch_size, all_moves=True)

            self.orders[current_index : current_index + batch_size] = orders
            self.movements[current_index : current_index + batch_size] = moves

        self.movements = one_hot(self.movements, num_classes=5)

    def __len__(self):
        return self.batches * self.batch_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        order = self.orders[idx]
        movement = self.movements[idx]

        image = self.data.get(order)
        images_moved = torch.zeros((4, 28 * 3, 28 * 3))
        for index, move in enumerate(movement):
            if torch.all(move[-1] != 1):
                zero_index = self.data.zero_index(order)
                new_order, _ = self.data.move(
                    order, zero_index, LABEL_TO_MOVEMENT[index]
                )
                images_moved[index] = self.data.get(new_order)

        return image, images_moved, movement
