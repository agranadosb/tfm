from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor

from tfm.constants.types import LightsSample
from tfm.data.base import BaseGenerator


class LightsOutGenerator(BaseGenerator):
    """
    This class manages the lights out dataset generation. A lights-out problem
    is a `nxn` grid of lights, where each light can be on or off. The goal is to
    turn off all the lights. The player can press a light, which will toggle
    the state of that light and its neighbors. There are n * n lights, so there
    are n * n possible actions (one for each light).

    A state of the game is represented as a nxn tensor of 0s and 1s, where 0
    means the light is off and 1 means the light is on. Each sample of this
    problem is a tuple of a state and a sequence of states. The sequence of
    states is a list which length is the number of lights. Each element
    of the sequence is the result of changing the state of the light on the
    index of the element of the sequence.

    For example, suppose a 3x3 grid of lights. A state is:

    >>> torch.ones(3, 3)

    Then, the Sample associated to this state is:

    >>> sample = (
    ...     torch.ones(3, 3),
    ...     (
    ...         torch.tensor([[False, False, True], [False, False, True], [True, True, True]]),
    ...         torch.tensor([[False, False, False], [False, False, False], [True, True, True]]),
    ...         torch.tensor([[True, False, False], [True, False, False], [True, True, True]]),
    ...         torch.tensor([[False, False, True], [False, False, True], [False, False, True]]),
    ...         torch.tensor([[False, False, False], [False, False, False], [False, False, False]]),
    ...         torch.tensor([[True, False, False], [True, False, False], [True, False, False]]),
    ...         torch.tensor([[True, True, True], [False, False, True], [False, False, True]]),
    ...         torch.tensor([[True, True, True], [False, False, False], [False, False, False]]),
    ...         torch.tensor([[True, True, True], [True, False, False], [True, False, False]]),
    ...     )
    ... )

    As we can see, there is a sequence of 9 states. This is because there is
    9 lights and each light can be turned on or off (affecting the neighbors).

    Each State can be represented as a matrix. For example, the state:

    >>> torch.tensor([[False, False, True], [False, False, True], [True, True, True]])

    Can be represented as:

    >>> # 0 0 1
    >>> # 0 0 1
    >>> # 1 1 1

    Another example on matrix format. Suppose a 3x3 grid of lights. A source
    state is:

    >>> # 1 0 1
    >>> # 0 1 0
    >>> # 1 0 1

    Then, all the possible new states are:

    >>> # 0 1 1   0 1 0   1 1 0
    >>> # 1 0 0   1 0 1   0 0 1
    >>> # 1 0 1   1 0 1   1 0 1

    >>> # 0 1 1   0 1 0   1 1 0
    >>> # 1 0 0   1 0 1   0 0 1
    >>> # 0 1 1   0 1 0   1 1 0

    >>> # 0 1 1   1 0 1   1 0 1
    >>> # 1 0 0   1 0 1   0 0 1
    >>> # 0 1 1   0 1 0   1 1 0

    Each action is represented as an integer. For example, the action 0 means
    change the state of the light 0, the action 1 means change the state of the
    light 1, and so on. As an example, the next list represents the actions
    that change the state of the previous example:

    >>> [ 0, 1, 2, 3, 4, 5, 6, 7, 8 ]

    On this problem there is not any not valid action. All the actions are
    valid.

    This class implements the BaseGenerator interface, so it can be used to
    generate data for the 8-puzzle problem. To know more about how to use the
    BaseGenerator interface, check the documentation of the BaseGenerator
    class.

    Examples
    --------
    >>> generator = LightsOutGenerator(1, 1, n=3)
    >>> generator.sequence()
    [(
        torch.tensor([[True, True, True], [True, True, True], [True, True, True]]),
        (
            torch.tensor([[False, False, True], [False, False, True], [True, True, True]]),
            torch.tensor([[False, False, False], [False, False, False], [True, True, True]]),
            torch.tensor([[True, False, False], [True, False, False], [True, True, True]]),
            torch.tensor([[False, False, True], [False, False, True], [False, False, True]]),
            torch.tensor([[False, False, False], [False, False, False], [False, False, False]]),
            torch.tensor([[True, False, False], [True, False, False], [True, False, False]]),
            torch.tensor([[True, True, True], [False, False, True], [False, False, True]]),
            torch.tensor([[True, True, True], [False, False, False], [False, False, False]]),
            torch.tensor([[True, True, True], [True, False, False], [True, False, False]]),
        )
    )]

    Parameters
    ----------
    sequences : int
        Number of sequences to generate.
    sequence_length : int
        Length of the sequences to generate.
    n : int
        The size of the grid.
    shuffle : bool, optional
        If True, the sequences are shuffled, by default True.
    size : int = 32
        The size of a light. Default is 32.
    line_value: int = 128
        The value of the line between lights. Default is 128.
    line_length: int = 4
        The length of the line between lights. Default is 4.
    """

    def __init__(
        self,
        sequences: int,
        sequence_length: int,
        /,
        *,
        n: int,
        shuffle: bool = True,
        size: int = 32,
        line_value: int = 128,
        line_length: int = 4,
    ):
        super().__init__(sequences, sequence_length, n * n, shuffle)
        self.n = n
        self.size = size - line_length * 2
        self.line_value = line_value
        self.line_length = line_length

    def init_state(self) -> Tensor:
        """
        Returns the initial state of the game. The initial state is a tensor of
        size (n * n) of 1s. For example, if n is 3, then the initial state is:

        >>> torch.tensor([[True, True, True, True, True, True, True, True, True]])

        This is represented in matrix format as:

        >>> # 1 1 1
        >>> # 1 1 1
        >>> # 1 1 1

        Examples
        --------
        >>> generator = LightsOutGenerator(1, 2, n=3).init_state()
        tensor([[True, True, True, True, True, True, True, True, True]])

        Returns
        -------
        Tensor
            The initial state of the game in a tensor of size (n * n).
        """
        return torch.ones(self.n * self.n, dtype=torch.bool)

    def select(self, sample: LightsSample) -> Tensor:
        # noinspection PyShadowingNames,PyTypeChecker,PyUnresolvedReferences
        """
        Given a LightsSample, it selects a Tensor from the sequence of Tensor of
        the LightsSample.

        Parameters
        ----------
        sample : Sample

        Examples
        --------
        >>> generator = LightsOutGenerator(1, 2, n=3)
        >>> sample = (
        ...     torch.ones(3, 3),
        ...     (
        ...         torch.tensor([[False, False, True], [False, False, True], [True, True, True]]),
        ...         torch.tensor([[False, False, False], [False, False, False], [True, True, True]]),
        ...         torch.tensor([[True, False, False], [True, False, False], [True, True, True]]),
        ...         torch.tensor([[False, False, True], [False, False, True], [False, False, True]]),
        ...         torch.tensor([[False, False, False], [False, False, False], [False, False, False]]),
        ...         torch.tensor([[True, False, False], [True, False, False], [True, False, False]]),
        ...         torch.tensor([[True, True, True], [False, False, True], [False, False, True]]),
        ...         torch.tensor([[True, True, True], [False, False, False], [False, False, False]]),
        ...         torch.tensor([[True, True, True], [True, False, False], [True, False, False]]),
        ...     )
        ... )
        >>> generator.select(sample)
        torch.tensor([[False, False, True], [False, False, True], [False, False, True]])

        Returns
        -------
        State
            The selected Tensor.
        """
        non_none = [s for s in sample[1] if s is not None]
        return non_none[np.random.randint(len(non_none))]

    def move(self, state: Tensor, action: int) -> Tensor:
        """
        Applies the given action to the given state. The result is a tensor of
        size (n * n) of the new state. The action just flips the light and the
        lights that are on the direct neighborhood.

        Parameters
        ----------
        state: torch.Tensor
            The state of the game.
        action: int
            The action to apply. It is the index of the light to flip.

        Examples
        --------
        Suppose a 3x3 grid of lights and a state:

        >>> # 1 1 1
        >>> # 1 1 1
        >>> # 1 1 1

        Then, the action 1 flips the light 1 (that is the second light) and the
        lights on the direct neighborhood:

        >>> # 0 0 0
        >>> # 0 0 0
        >>> # 1 1 1

        Returns
        -------
        Tensor
            The new state with shape (n * n).
        """
        state = state.clone().reshape(self.n, self.n)
        new_state = state.clone()
        i, j = divmod(action, self.n)
        top_x = i - 1
        top_y = j - 1
        bottom_x = i + 2
        bottom_y = j + 2

        if top_x < 0:
            top_x = 0
        if top_y < 0:
            top_y = 0
        if bottom_x > self.n * self.n:
            bottom_x = self.n * self.n
        if bottom_y > self.n * self.n:
            bottom_y = self.n * self.n

        new_state[top_x:bottom_x, top_y:bottom_y] = ~state[
            top_x:bottom_x, top_y:bottom_y
        ]

        return new_state.reshape(self.n * self.n)

    def _compute_border_coordinate(
        self, x: int, y: int, border_type: str
    ) -> Tuple[int, int]:
        """
        Given a coordinate, computes the coordinate of the border of a light.
        If the border type is "x", then the border is computed over the x-axis
        and the coordinate of the border is (x, y). If the border type is
        "y", then the border is computed over the y-axis and the coordinate of
        the border is (y, x).

        Parameters
        ----------
        x: int
            X coordinate.
        y: int
            Y coordinate.
        border_type: str
            The type of border. It can be "x" that means the border is computed
            over the x-axis or "y" that means the border is computed over the
            y-axis.

        Returns
        -------
        Tuple[int, int]
            A coordinate of the border.
        """
        if border_type == "x":
            return x, y
        return y, x

    def _compute_border(
        self, border_type: str, base: int, init: int, fin: int, value: int = 1
    ) -> List[Tuple[int, int]]:
        """
        Computes the coordinates of the border of a light.

        Parameters
        ----------
        border_type: str
            The type of border. It can be "x" that means the border is computed
            over the x-axis or "y" that means the border is computed over the
            y-axis.
        base: int
            The base coordinate.
        init: int
            The initial coordinate.
        fin: int
            The final coordinate.
        value: int = 1
            In which direction the border is computed. If value is 1, then the
            border is computed from top to bottom. If value is -1, then the
            border is computed from bottom to top. Default is 1.

        Examples
        --------
        Suppose a 2x2 grid of lights, a size of 32 and a border length of 2.
        The base for the first light is 0. The initial coordinate is 0 and the
        final coordinate is 32. Then, the result is:

        ```
        (0, 0) (0, 1) (0, 2) (0, 3)
        (0, 4) (0, 5) (0, 6) (0, 7)
        ...
        (0, 28) (0, 29) (0, 30) (0, 31)
        (1, 0) (1, 1) (1, 2) (1, 3) # The length of the border is 2, so it's
        (1, 4) (1, 5) (1, 6) (1, 7) # necessary to add draw 2 lines.
        ...
        (1, 28) (1, 29) (1, 30) (1, 31)
        ```

        Which is the top border of the first light.

        If the light is the second one, then the base is 0. The initial
        coordinate is 32 and the final coordinate is 64.

        The same applies for the bottom border, but the base is 63.

        Returns
        -------
        List[Tuple[int, int]]
            The coordinates of the y border of a light.
        """
        result = []
        for i in range(self.line_length):
            result += [
                self._compute_border_coordinate(base + i * value, index, border_type)
                for index in range(init, fin)
            ]
        return result

    def image(self, state: Tensor) -> Tensor:
        """
        Returns the image of the state. The image is a tensor of size
        (n * (size + 2), n * (size + 2)).

        Parameters
        ----------
        state: Tensor
            The state of the game.


        Examples
        --------

        Suppose a 2x2 grid of lights and a size of 30. A state is:

        >>> # 1 1
        >>> # 1 0

        Then, the result is a tensor of size (64, 64) that represents the image
        where 3 lights are on and 1 light is off.

        Returns
        -------
        Tensor
            The image of the state in a tensor of size
            (n * (size + 2), n * (size + 2)).
        """
        square_size = self.size + self.line_length * 2
        image = torch.zeros(
            self.n * square_size, self.n * square_size, dtype=torch.uint8
        )
        for i in range(self.n):
            for j in range(self.n):
                # Get index on the vector state
                index = i * self.n + j
                # Light square coordinates
                top_x = i * square_size
                top_y = j * square_size
                bottom_x = top_x + square_size
                bottom_y = top_y + square_size

                # Populate light if it is on
                if state[index] == 1:
                    image[top_x:bottom_x, top_y:bottom_y] = 255

                # Create border
                total_coordinates = (
                    self._compute_border("y", top_y, top_x, bottom_x)
                    + self._compute_border("y", bottom_y - 1, top_x, bottom_x, -1)
                    + self._compute_border("x", top_x, top_y, bottom_y)
                    + self._compute_border("x", bottom_x - 1, top_y, bottom_y, -1)
                )

                x_coordinates, y_coordinates = zip(*total_coordinates)

                # Populate border
                image[x_coordinates, y_coordinates] = self.line_value
        return image.unsqueeze(0)
