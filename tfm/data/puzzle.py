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
    """This class is used to manage the generation of 8-puzzle data. The class uses
    "orders" to manage the position of the digits in the grid and to apply
    movements to the grid.

    For example, the next order:

    ```
    [ 1, 2, 3, 8, 0, 4, 7, 6, 5]
    ```

    Represents the next grid:

    ```
    1 2 3
    8 0 4
    7 6 5
    ```

    If we want to apply a random movement to the grid, we can use the method
    `random_move`:

    ```
    order = [ 1, 2, 3, 8, 0, 4, 7, 6, 5]
    zero_index = 4
    generator = Puzzle8MnistGenerator(order)

    new_order, new_zero_index, movement = generator.random_move(order, zero_index)
    ```

    A possible result of this code is:

    ```
    new_order = [ 1, 2, 3, 8, 4, 0, 7, 6, 5]
    new_zero_index = 5
    movement = 1
    ```

    And the new grid is:

    ```
    1 2 3
    8 4 0
    7 6 5
    ```

    So the movement is `1` because the zero digit has moved to the right.

    When we want to transform an order to an image in a tensor format, we can use
    the method `get`:

    ```
    order = [ 1, 2, 3, 8, 0, 4, 7, 6, 5]
    generator = Puzzle8MnistGenerator(order)

    image = generator.get(order)
    ```

    The image will be a tensor of size `(28, 28)` which represents the grid:

    ```
    1 2 3
    8 0 4
    7 6 5
    ```

    This class more functions to manage random sequences, getting all possible
    movements from a grid, etc.

    The movements can be represented as indices or movement applied to an
    index:

     - Right movement   :  1 or 0
     - Left movement    : -1 or 1
     - Bottom movement  : -3 or 2
     - Top movement     :  3 or 3

    Parameters
    ----------
    order: List[int] = [1, 2, 3, 8, 0, 4, 7, 6, 5]
        Default order on the grid. The default order is:
        ```
        1 2 3
        8 0 4
        7 6 5
        ```
    """

    def __init__(self, order: List[int] = None):
        if order is None:
            order = ORDERED_ORDER

        self.size = 28 * 3
        self.dataset = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transforms.ToTensor()
        )
        self.order = torch.IntTensor(order).to(torch.int8)

        if not has_correct_order(order):
            raise ValueError(
                "The list must have only the next values [0, 1, 2, 3, 4, 5, 6, 7, 8]"
            )

        self.digits_indices = {
            index: np.zeros(10, dtype=np.int16) for index in range(10)
        }
        digits_counter = {index: -1 for index in range(10)}
        completed = np.zeros(10, dtype=bool)
        for index in range(len(self.dataset)):
            _, digit = self.dataset[index]

            completed[digit] = completed[digit] or digits_counter[digit] == 9
            current_digit_list_index = (digits_counter[digit] + 1) % 10

            digits_counter[digit] = current_digit_list_index
            self.digits_indices[digit][current_digit_list_index] = index

            if completed.all():
                break

    def _is_beyond_bounds(self, index: int, movement: int) -> bool:  # noqa
        """
        Check if the index is beyond the bounds of the grid.

        An example of a movement applied to an index that is beyond the bounds
        of the grid is when the index is 0 and the movement is -1.

        An example that is into the bounds is when the index is 8 and the
        movement is 1

        ```
        generator = Puzzle8MnistGenerator()

        result = generator._is_beyond_bounds(0, -1)
        print(result) # True

        result = generator._is_beyond_bounds(8, 1)
        print(result) # False
        ```

        Parameters
        ----------
        index: int
            Index to check.
        movement: int
            Movement to check as index shift.

        Returns
        -------
        bool
            True if the index is beyond the bounds of the grid, False otherwise.
        """
        new_index = index + movement
        return new_index < 0 or new_index > 8

    def _is_incorrect_left(self, index: int, movement: int) -> bool:  # noqa
        """
        Check if the movement to the left is incorrect.

        An example of a movement to the left that is incorrect is when the index
        is 3 and the movement is -1.

        An example of a movement to the left that is correct is when the index
        is 1 and the movement is -1.

        ```
        generator = Puzzle8MnistGenerator()

        result = generator._is_incorrect_left(3, -1)
        print(result) # True

        result = generator._is_incorrect_left(1, -1)
        print(result) # False
        ```

        Parameters
        ----------
        index: int
            Index of the zero digit.
        movement: int
            Movement to check as index shift.

        Returns
        -------
        bool
            True if the movement to the left is incorrect, False otherwise.
        """
        return index % 3 == 0 and movement == -1

    def _is_incorrect_right(self, index: int, movement: int) -> bool:  # noqa
        """
        Check if the movement to the right is incorrect.

        An example of a movement to the right that is incorrect is when the index
        is 2 and the movement is 1.

        An example of a movement to the right that is correct is when the index
        is 1 and the movement is 1.

        ```
        generator = Puzzle8MnistGenerator()

        result = generator._is_incorrect_right(2, 1)
        print(result) # True

        result = generator._is_incorrect_right(1, 1)
        print(result) # False
        ```

        Parameters
        ----------
        index: int
            Index of the zero digit.
        movement: int
            Movement to check as index shift.

        Returns
        -------
        bool
            True if the movement to the right is incorrect, False otherwise.
        """
        return index % 3 == 2 and movement == 1

    def is_possible(self, zero_index: int, movement: int) -> bool:
        """
        Check if the movement is possible. A movement is possible if is not
        beyond the bounds of the grid, is not incorrect to the left and is not
        incorrect to the right.

        An example of a movement that is not possible is when the index is 0 and
        the movement is -1.

        An example of a movement that is possible is when the index is 8 and the
        movement is -3.

        ```
        generator = Puzzle8MnistGenerator()

        result = generator.is_possible(0, -1)
        print(result) # False

        result = generator.is_possible(8, -3)
        print(result) # True
        ```

        Parameters
        ----------
        zero_index: int
            Index of the zero digit.
        movement: int
            Movement to check as index shift.

        Returns
        -------
        bool
            True if the movement is possible, False otherwise.
        """
        return not (
            self._is_beyond_bounds(zero_index, movement)
            or self._is_incorrect_left(zero_index, movement)
            or self._is_incorrect_right(zero_index, movement)
        )

    def zero_index(self, order: Tensor) -> int:  # noqa
        """
        Get the index of the zero digit.

        An example of use is:

        ```
        generator = Puzzle8MnistGenerator()

        order = torch.tensor([ 1, 2, 3, 8, 0, 4, 7, 6, 5])
        zero_index = generator.zero_index(order)

        print(zero_index) # 4
        ```

        Parameters
        ----------
        order: Tensor
            Order of the grid.

        Returns
        -------
        int
            Index of the zero digit.
        """
        return np.where(order == 0)[0][0].item()

    def move(self, order: Tensor, movement: int) -> Tensor:
        """
        Move the zero digit on the order based on the movement.

        An example of use is:

        ```
        order = [ 1, 2, 3, 8, 0, 4, 7, 6, 5]
        generator = Puzzle8MnistGenerator(order)

        new_order, new_zero_index = generator.move(order, 1)

        print(new_order) # [ 1, 2, 3, 8, 4, 0, 7, 6, 5]

        # As a grid
        # 1 2 3
        # 8 4 0
        # 7 6 5
        ```

        Parameters
        ----------
        order: Tensor
            Order of the grid.
        movement: int
            Movement to check as index shift.

        Raises
        ------
        ValueError
            If the movement is not possible.

        Returns
        -------
        Tensor
            New order.
        """
        zero_index = self.zero_index(order)
        if not self.is_possible(zero_index, movement):
            raise ValueError("The movement is not possible")

        return self._move(order, zero_index, movement)[0]

    def _move(  # noqa
        self, order: Tensor, zero_index: int, movement: int
    ) -> Tuple[Tensor, int]:
        """
        Move the zero digit on the order based on the movement.

        An example of use is:

        ```
        order = [ 1, 2, 3, 8, 0, 4, 7, 6, 5]
        generator = Puzzle8MnistGenerator(order)

        new_order, new_zero_index = generator.move(order, 4, 1)

        print(new_order) # [ 1, 2, 3, 8, 4, 0, 7, 6, 5]

        # As a grid
        # 1 2 3
        # 8 4 0
        # 7 6 5
        ```

        Parameters
        ----------
        order:
            Order of the grid.
        zero_index: int
            Index of the zero digit.
        movement:
            Movement to check as index shift.

        Returns
        -------
        Tuple[Tensor, int]
            New order and new index of the zero digit.
        """
        new_index = zero_index + movement

        new_order = order.clone()
        new_order[zero_index] = new_order[new_index]
        new_order[new_index] = 0

        return new_order, new_index

    def _all_moves(self, order: Tensor, zero_index: int) -> Tuple[Tensor, int, Tensor]:
        """
        Get all the possible moves from the current order.

        An example of use is:

        ```
        generator = Puzzle8MnistGenerator()

        order = torch.tensor([ 1, 0, 3, 8, 2, 4, 7, 6, 5])
        zero_index = generator.zero_index(order)

        new_order, new_zero_index, movement_label, movements_labels = generator._all_moves(order, zero_index)

        print(new_order) # [ 1, 2, 3, 8, 0, 4, 7, 6, 5] random movement which is possible
        print(new_zero_index) # 4
        print(movement_label) # 3
        print(movements_labels) # [0, 1, 4, 3] The four shows that the up movement is not possible
        ```

        Parameters
        ----------
        order: Tensor
            Order of the grid.
        zero_index: int
            Index of the zero digit.

        Returns
        -------
        Tuple[Tensor, int, Tensor]
            New order, new index of the zero digit and labels of the movements
            or 4 if the movement is not possible.
        """
        movements_labels = torch.full((4,), 4, dtype=torch.int64)
        movements = movements_labels.clone()

        for index, movement in enumerate(MOVEMENTS):
            if self.is_possible(zero_index, movement):
                movements_labels[index] = MOVEMENT_TO_LABEL[movement]
                movements[index] = movement

        movement_to_apply = np.random.choice(movements[movements != 4])
        order, zero_index = self._move(order, zero_index, movement_to_apply)

        return (
            order,
            zero_index,
            movements_labels,
        )

    def random_move(
        self, current_order: Tensor, zero_index: int
    ) -> Tuple[Tensor, int, int]:
        """
        Get a random move from the current order.

        An example of use is:

        ```
        generator = Puzzle8MnistGenerator()

        order = torch.tensor([ 1, 0, 3, 8, 2, 4, 7, 6, 5])
        zero_index = generator.zero_index(order)

        new_order, new_zero_index, movement_label = generator.random_move(order, zero_index)

        print(new_order) # [ 1, 2, 3, 8, 0, 4, 7, 6, 5] random movement which is possible
        print(new_zero_index) # 4
        print(movement_label) # 3
        ```

        Parameters
        ----------
        current_order: Tensor
            Order of the grid.
        zero_index: int
            Index of the zero digit.

        Returns
        -------
        Tuple[Tensor, int, int]
            New order, new index of the zero digit and label of the movement.
        """
        movement = random.choice(MOVEMENTS)
        if not self.is_possible(zero_index, movement):
            movement = -1 * movement
        new_order, new_index = self._move(current_order, zero_index, movement)
        return new_order, new_index, MOVEMENT_TO_LABEL[movement]

    def random_sequence(
        self, size: int, all_moves: bool = False
    ) -> Tuple[Tensor, Tensor]:
        """
        Get a random sequence of unordered moves from the original order.

        An example of use is:

        ```
        generator = Puzzle8MnistGenerator()

        orders, movements = generator.random_sequence(10)

        print(orders) # [[ 1, 2, 3, 8, 0, 4, 7, 6, 5], [ 1, 2, 3, 8, 4, 0, 7, 6, 5], ...]
        print(movements) # [3, 4, ...]
        ```

        Parameters
        ----------
        size: int
            Size of the sequence.
        all_moves: bool
            If True, all the possible moves are returned, otherwise a random
            move is returned.

        Returns
        -------
        Tuple[Tensor, Tensor]
            Sequence of orders and sequence of movements.
        """
        order = self.order.clone()
        zero_index = self.zero_index(order)

        total_size = size * 4
        moves_size = total_size
        method = self.random_move
        if all_moves:
            method = self._all_moves
            moves_size = (total_size, 4)

        orders = torch.zeros((total_size, 9), dtype=torch.int8)
        movements = torch.zeros(moves_size, dtype=torch.int64)

        for i in range(total_size):
            orders[i] = order
            order, zero_index, movement = method(order, zero_index)
            movements[i] = movement

        choices = np.random.choice(np.arange(0, total_size), size)

        return orders[choices], movements[choices]

    def get(self, order: Sequence) -> torch.Tensor:
        """Returns the 8-puzzle wrote on 'sequence'.

        An example of use is:

        ```
        generator = Puzzle8MnistGenerator()

        order = [ 1, 2, 3, 8, 0, 4, 7, 6, 5]
        image = generator.get(order)

        print(image.shape) # torch.Size([28, 28])
        ```

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

                image, _ = self.dataset[np.random.choice(self.digits_indices[digit])]

                base_image[xmin:xmax, ymin:ymax] = image

        return base_image


class Puzzle8MnistDataset(Dataset):
    """
    Dataset of 8-Puzzle images.

    An example of use is:

    ```
    dataset = Puzzle8MnistDataset(16, 32)

    print(len(dataset)) # 512 = 16 * 32

    image, moved_image, movements = dataset[0]

    print(image.shape) # torch.Size([32, 1, 28, 28])
    print(moved_image.shape) # torch.Size([32, 1, 28, 28])
    print(movements.shape) # torch.Size([32, 5])
    ```

    Parameters
    ----------
    batches: int
        Number of batches.
    batch_size: int
        Size of each batch.
    """

    def __init__(self, batches: int, batch_size: int):
        self._length = batches * batch_size

        self.data = Puzzle8MnistGenerator(order=[1, 2, 3, 8, 0, 4, 7, 6, 5])

        self.orders = torch.zeros((self._length, 9), dtype=torch.int8)
        movements = torch.zeros((self._length, 4), dtype=torch.int64)
        for i in range(batches):
            current_index = i * batch_size
            orders, moves = self.data.random_sequence(batch_size, all_moves=True)

            self.orders[current_index: current_index + batch_size] = orders
            movements[current_index: current_index + batch_size] = moves

        self.movements = one_hot(movements, num_classes=5)

    def __len__(self):
        return self._length

    def _apply_movement(self, order: Tensor, movements: Tensor) -> Tensor:
        """
        Apply the movement to the order.

        Parameters
        ----------
        order: Tensor
            Order of the grid.
        movements:
            Movements to apply.

        Returns
        -------
        Tensor
            New order.
        """
        images_moved = torch.zeros((4, self.data.size, self.data.size))
        for index, move in enumerate(movements):
            if torch.any(move[-1] == 1):
                images_moved[index] = self.data.get(
                    self.data.move(order, LABEL_TO_MOVEMENT[index])
                )
        return images_moved

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        order = self.orders[idx]
        movement = self.movements[idx]

        image = self.data.get(order)
        images_moved = self._apply_movement(order, movement)

        return image, images_moved, movement
