from typing import List, Tuple

import torch


class LightsOutGenerator:
    """
    This class manages the lights out dataset generation. A lights-out problem
    is a nxn grid of lights, where each light can be on or off. The goal is to
    turn off all the lights. The player can press a light, which will toggle
    the state of that light and its neighbors.

    A state of the game is represented as a nxn matrix of 0s and 1s, where 0
    means the light is off and 1 means the light is on. Each sample is composed
    by a source state, a list of new states and a list of actions.

    The source is a tensor which dimensions are (n, n). The new states are a
    tensor which dimensions are (n * n, n, n). The actions are a tensor which
    dimensions are (n * n, n * n + 1).

    There are n * n lights, so there are n * n possible actions (one for each
    light). The last action is the "not-possible" action, which means that the
    action is not possible. In this case there is not a "not-possible" action,
    but it is added to maintain consistency with other datasets.

    Examples
    --------

    Suppose a 3x3 grid of lights. A source state is:

    ```
    1 0 1
    0 1 0
    1 0 1
    ```

    Then, all the possible new states are:

    ```
    0 1 1   0 1 0   1 1 0
    1 0 0   1 0 1   0 0 1
    1 0 1   1 0 1   1 0 1

    0 1 1   0 1 0   1 1 0
    1 0 0   1 0 1   0 0 1
    0 1 1   0 1 0   1 1 0

    0 1 1   1 0 1   1 0 1
    1 0 0   1 0 1   0 0 1
    0 1 1   0 1 0   1 1 0
    ```

    And the actions are:

    ```
    1 0 0 0 0 0 0 0 0 0
    0 1 0 0 0 0 0 0 0 0
    0 0 1 0 0 0 0 0 0 0
    0 0 0 1 0 0 0 0 0 0
    0 0 0 0 1 0 0 0 0 0
    0 0 0 0 0 1 0 0 0 0
    0 0 0 0 0 0 1 0 0 0
    0 0 0 0 0 0 0 1 0 0
    0 0 0 0 0 0 0 0 1 0
    ```

    This class will provide methods to return an image given a state, to
    provide a random sequence of states from the initial state and to generate
    all possible states from a given state.

    The image representation of a state is a tensor of size (n * (size + 2), n * (size + 2))
    where size is the size of a light. The light is represented as a square of
    size (size, size) and the lights are separated by a line of size 1,
    so the total size of a light is (size + 2, size + 2).

    Parameters
    ----------
    n : int
        The size of the grid.
    size : int = 31
        The size of a light. Default is 31.
    line_value: int = 128
        The value of the line between lights. Default is 128.
    """

    def __init__(self, n: int, size: int = 30, line_value: int = 128):
        self.n = n
        self.size = size
        self.line_value = line_value

    def _compute_x_border(
        self, base: int, init: int, fin: int
    ) -> List[Tuple[int, int]]:
        """
        Computes the coordinates of the x border of a light. The x border can
        be the top or the bottom border.

        Parameters
        ----------
        base: int
            The base coordinate.
        init: int
            The initial coordinate.
        fin: int
            The final coordinate.

        Examples
        --------
        Suppose a 2x2 grid of lights and a size of 30. The base for the first
        light is 0. The initial coordinate is 0 and the final coordinate
        is 32. Then, the result is:

        ```
        (0, 0) (0, 1) (0, 2) (0, 3)
        (0, 4) (0, 5) (0, 6) (0, 7)
        ...
        (0, 28) (0, 29) (0, 30) (0, 31)
        ```

        Which is the top border of the first light.

        If the light is the second one, then the base is 0. The initial
        coordinate is 32 and the final coordinate is 64.

        The same applies for the bottom border, but the base is 63.

        Returns
        -------
        List[Tuple[int, int]]
            The coordinates of the x border of a light.
        """
        return [
            (index, base) for index in range(init, fin)
        ]

    def _compute_y_border(self, base: int, init: int, fin: int) -> List[Tuple[int, int]]:
        """
        Computes the coordinates of the y border of a light. The y border can
        be the left or the right border.

        Parameters
        ----------
        base: int
            The base coordinate.
        init: int
            The initial coordinate.
        fin: int
            The final coordinate.

        Examples
        --------
        The same applies as in the `_compute_x_border` method but instead of
        taking y as the base coordinate, it takes x.

        Returns
        -------
        List[Tuple[int, int]]
            The coordinates of the y border of a light.
        """
        return [
            (base, index) for index in range(init, fin + 1)
        ]

    def get(self, state: torch.Tensor) -> torch.Tensor:
        """
        Returns the image of the state. The image is a tensor of size (n * (size + 2), n * (size + 2)).

        Parameters
        ----------
        state: torch.Tensor
            The state of the game.


        Examples
        --------

        Suppose a 2x2 grid of lights and a size of 30. A state is:

        ```
        1 1
        1 0
        ```

        Then, the result is a tensor of size (64, 64) that represents the image
        where 3 lights are on and 1 light is off.

        Returns
        -------
        torch.Tensor
            The image of the state.
        """
        image = torch.zeros(self.n * (self.size + 2), self.n * (self.size + 2))
        for i in range(self.n):
            for j in range(self.n):
                # Get index on the vector state
                index = i * self.n + j
                # Light square coordinates
                top_x = i * (self.size + 2)
                top_y = j * (self.size + 2)
                bottom_x = top_x + self.size + 1
                bottom_y = top_y + self.size + 1

                # Populate light if it is on
                if state[index] == 1:
                    image[top_x:bottom_x, top_y:bottom_y] = 255

                # Create border
                total_coordinates = (
                    self._compute_x_border(top_y, top_x, bottom_x) +
                    self._compute_x_border(bottom_y, top_x, bottom_x) +
                    self._compute_y_border(top_x, top_y, bottom_y) +
                    self._compute_y_border(bottom_x, top_y, bottom_y)
                )

                x_coordinates, y_coordinates = zip(*total_coordinates)

                # Populate border
                image[x_coordinates, y_coordinates] = self.line_value
        return image

    def random_sequence(self, length: int) -> torch.Tensor:
        """
        Returns a random sequence of states of the given length. The result is
        a tensor of size (length, n, n) of random states.

        Parameters
        ----------
        length: int
            The length of the sequence.

        Examples
        --------

        Suppose a 2x2 grid of lights and a length of 3. A result could be:

        ```
        1 1   1 0   0 0
        1 1   0 0   0 0
        ```

        That is a tensor of size (3, 2, 2) of random states.

        Returns
        -------
        torch.Tensor
            A random sequence of states.
        """
        pass

    def all_states(self, state: torch.Tensor) -> torch.Tensor:
        """
        Returns all possible states from the given state. The result is a
        tensor of size (n * n, n, n) of all possible states.

        Parameters
        ----------
        state: torch.Tensor
            The state of the game.

        Examples
        --------

        Suppose a 3x3 grid of lights. A state is:

        ```
        1 1 1
        1 1 1
        1 1 1
        ```

        Then, the result is a tensor of size (9, 3, 3) of all possible states:

        ```
        0 0 1   0 0 0   1 0 0
        0 0 1   0 0 0   1 0 0
        1 1 1   1 1 1   1 1 1

        0 0 1   0 0 0   1 0 0
        0 0 1   0 0 0   1 0 0
        0 0 1   0 0 0   1 0 0

        1 1 1   1 1 1   1 1 1
        0 0 1   0 0 0   1 0 0
        0 0 1   0 0 0   1 0 0
        ```

        Returns
        -------
        torch.Tensor
            All possible states from the given state.
        """
        pass
