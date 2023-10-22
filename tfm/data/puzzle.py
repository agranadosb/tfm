import numpy as np
import torch
import torchvision
from torch import Tensor
from torchvision import transforms

from tfm.constants.app import ROOT
from tfm.constants.puzzle import DEFAULT_ORDER, ACTION_TO_MOVEMENT
from tfm.constants.types import TensorSample
from tfm.data.base import BaseGenerator
from tfm.utils.puzzle import has_correct_order


class Puzzle8MnistGenerator(BaseGenerator):
    # noinspection PyShadowingNames,PyTypeChecker,PyUnresolvedReferences
    """This class is used to manage the generation of 8-puzzle data. The 8-puzzle
    problem is based on ordering a grid of 3x3 with 8 digits and a blank space.
    The digits are from 1 to 8 and the blank space is represented by 0. The
    blank space can be moved to the right, left, top or bottom. The goal is to
    order the grid only with the movements of the blank space. The order that
    we want to achieve is:

    >>> # 1 2 3
    >>> # 8 0 4
    >>> # 7 6 5

    The grid is represented as a list of 9 digits. The tensor representation of
    this list will be the state of the problem. For example, the previous
    grid is represented as:

    >>> torch.tensor([1, 2, 3, 8, 0, 4, 7, 6, 5], dtype=torch.int8)

    A movement is represented as an index shift. For example, if we want to move
    the blank space to the right, we need to shift the index of the blank space
    to the right. In the previous example, the index of the blank space is 4.
    If we move the blank space to the right, the new index of the blank space
    is 5. The movement is represented as the shift applied to the index where
    the blank space is located. In this case, the movement is 1:

    >>> [ 1, 2, 3, 8, 0, 4, 7, 6, 5] -> [ 1, 2, 3, 8, 4, 0, 7, 6, 5]

    As it can be seen, the blank space has been moved to the right and the
    digit 4 has been moved to the left (where the blank space was located).

    The total movements that can be applied to the blank space are 4: Top,
    Bottom, Left and Right. The movements are represented as indices shift:

    - Right movement   :  1
    - Left movement    : -1
    - Bottom movement  : -3
    - Top movement     :  3

    Each movement is also an action. The actions are represented as indices:

     - Right movement   :  0
     - Left movement    : -1
     - Bottom movement  : -2
     - Top movement     :  3

    So there is a direct mapping between the actions and the movements:

    >>> ACTION_TO_MOVEMENT = {0: 1, 1: -1, 2: 3, 3: -3}

    Each Sample of this problem is a tuple of two elements. The first element
    is the state of the grid and the second element is a tuple of 4 elements.
    Each element of the tuple is a possible state of the grid after applying
    the movement. If the movement is not possible, the element is None.

    This class implements the BaseGenerator interface, so it can be used to
    generate data for the 8-puzzle problem. To know more about how to use the
    BaseGenerator interface, check the documentation of the BaseGenerator
    class.

    Examples
    --------
    >>> generator = Puzzle8MnistGenerator(1, 2)
    >>> generator.sequence()
    [
        (
            torch.tensor([1, 2, 3, 8, 0, 4, 7, 6, 5], dtype=torch.int8),
            (
                torch.tensor([1, 2, 3, 8, 4, 0, 7, 6, 5], dtype=torch.int8),
                torch.tensor([1, 2, 3, 0, 8, 4, 7, 6, 5], dtype=torch.int8),
                torch.tensor([1, 2, 3, 8, 6, 4, 7, 0, 5], dtype=torch.int8)
                torch.tensor([1, 0, 3, 8, 2, 4, 7, 6, 5], dtype=torch.int8),
            ),
        ),
        (
            torch.tensor([1, 2, 3, 8, 6, 4, 7, 0, 5], dtype=torch.int8),
            (
                torch.tensor([1, 2, 3, 8, 6, 4, 7, 5, 0], dtype=torch.int8),
                torch.tensor([1, 2, 3, 8, 6, 4, 0, 7, 5], dtype=torch.int8),
                None,
                torch.tensor([1, 2, 3, 8, 0, 4, 7, 6, 5], dtype=torch.int8),
            ),
        )
    ]

    Parameters
    ----------
    sequences : int
        Number of sequences to generate.
    sequence_length : int
        Length of the sequences to generate.
    order: list[int] = [1, 2, 3, 8, 0, 4, 7, 6, 5]
        Default order on the grid and initial state of the grid.
    shuffle : bool, optional
        If True, the sequences are shuffled, by default True.
    """

    def __init__(
        self,
        sequences: int,
        sequence_length: int,
        /, *,
        order: list[int] = None,
        shuffle: bool = True,
    ):
        super().__init__(
            sequences, sequence_length, actions=4, shuffle=shuffle
        )
        if order is None:
            order = DEFAULT_ORDER

        self.dataset = torchvision.datasets.MNIST(
            root=ROOT,
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )

        if not has_correct_order(order):
            raise ValueError(
                "The list must have only the next values [0, 1, 2, 3, 4, 5, 6, 7, 8]"
            )
        self.order = torch.as_tensor(order, dtype=torch.int8)

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

    def init_state(self) -> Tensor:
        # noinspection PyShadowingNames,PyTypeChecker,PyUnresolvedReferences
        """
        Get the initial state of the grid. The initial state is the order of the
        grid.

        Examples
        --------
        >>> generator = Puzzle8MnistGenerator(1, 2)
        >>> generator.init_state()
        tensor([1, 2, 3, 8, 0, 4, 7, 6, 5], dtype=torch.int8)

        Returns
        -------
        Tensor
            Initial state of the grid.
        """
        return torch.as_tensor(self.order, dtype=torch.int8)

    def select(self, sample: TensorSample) -> Tensor:
        # noinspection PyShadowingNames,PyTypeChecker,PyUnresolvedReferences,GrazieInspection
        """
        Given a PuzzleSample, it selects a Tensor from the sequence of Tensor of
        the PuzzleSample.

        Parameters
        ----------
        sample : Sample

        Examples
        --------
        >>> generator = Puzzle8MnistGenerator(1, 2, n=3)
        >>> sample = (
        ...     torch.tensor([1, 2, 3, 8, 6, 4, 7, 0, 5], dtype=torch.int8]),
        ...     (
        ...         torch.tensor([1, 2, 3, 8, 6, 4, 7, 5, 0], dtype=torch.int8]),
        ...         torch.tensor([1, 2, 3, 8, 6, 4, 0, 7, 5], dtype=torch.int8]),
        ...         None,
        ...         torch.tensor([1, 2, 3, 8, 0, 4, 7, 6, 5], dtype=torch.int8]),
        ...     )
        ... )
        >>> generator.select(sample)
        torch.tensor([1, 2, 3, 8, 6, 4, 0, 7, 5], dtype=torch.int8)

        Returns
        -------
        State
            The selected Tensor.
        """
        non_none = [s for s in sample[1] if s is not None]
        return non_none[np.random.randint(len(non_none))]

    def is_possible(self, zero_index: int, movement: int) -> bool:
        # noinspection PyShadowingNames,PyTypeChecker,PyUnresolvedReferences
        """
        Check if the movement is possible. A movement is possible if is not
        beyond the bounds of the grid, is not incorrect to the left and is not
        incorrect to the right.

        An example of a movement that is not possible is when the index is 0 and
        the movement is -1.

        An example of a movement that is possible is when the index is 8 and the
        movement is -3.

        Examples
        --------
        >>> generator = Puzzle8MnistGenerator(1, 2)
        >>> result = generator.is_possible(0, -1)
        False

        >>> result = generator.is_possible(8, -3)
        True

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
        new_index = zero_index + movement
        is_beyond_bounds = new_index < 0 or new_index > 8
        is_incorrect_left = zero_index % 3 == 0 and movement == -1
        is_incorrect_right = zero_index % 3 == 2 and movement == 1

        return not (
            is_beyond_bounds or is_incorrect_left or is_incorrect_right
        )

    def move(self, state: Tensor, action: int) -> Tensor | None:
        # noinspection PyShadowingNames,PyTypeChecker,PyUnresolvedReferences
        """
        Move the zero digit on the order based on the movement. If the movement
        is not possible, None is returned.

        Examples
        --------
        >>> state = torch.tensor([ 1, 2, 3, 8, 6, 4, 7, 0, 5])
        >>> generator = Puzzle8MnistGenerator(1, 2)

        >>> generator.move(state, 1)
        tensor([ 1, 2, 3, 8, 6, 4, 7, 5, 0])

        >>> # As a grid
        >>> # 1 2 3
        >>> # 8 4 0
        >>> # 7 6 5

        Parameters
        ----------
        state: Tensor
            Order of the grid.
        action: int
            Index of the action to apply.

        Returns
        -------
        Tensor
            New order.
        """
        movement = ACTION_TO_MOVEMENT[action]

        zero_index = np.where(state == 0)[0][0].item()
        if not self.is_possible(zero_index, movement):
            return None

        new_index = zero_index + movement

        new_order = state.clone()
        new_order[zero_index] = new_order[new_index]
        new_order[new_index] = 0

        return new_order

    def image(self, state: Tensor) -> Tensor:
        # noinspection PyShadowingNames,PyTypeChecker,PyUnresolvedReferences
        """Returns the image of the grid based on the state.

        Example
        -------
        >>> generator = Puzzle8MnistGenerator(1, 2)
        >>> state = [ 1, 2, 3, 8, 0, 4, 7, 6, 5]
        >>> generator.image(state)
        tensor([[[0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000], ...], ...])

        Parameters
        ----------
        state: Tensor
            State of the grid.

        Returns
        -------
        Tensor
            Image of the grid.
        """
        base_image = torch.zeros((28 * 3, 28 * 3))
        for column in range(3):
            for row in range(3):
                ymin = column * 28
                xmin = row * 28
                ymax = ymin + 28
                xmax = xmin + 28

                index = row * 3 + column
                digit = state[index].item()

                image, _ = self.dataset[np.random.choice(self.digits_indices[digit])]

                base_image[xmin:xmax, ymin:ymax] = image

        return base_image.unsqueeze(0)
