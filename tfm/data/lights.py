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

    The image representation of a state is a tensor of size (n * size, n * size)
    where size is the size of a light. The light is represented as a square of
    size (size, size) and the lights are separated by a line of size 1,
    so the total size of a light is (size, size).

    Parameters
    ----------
    n : int
        The size of the grid.
    size : int = 31
        The size of a light. Default is 31.
    line_value: int = 128
        The value of the line between lights. Default is 128.
    """

    def __init__(self, n: int, size: int = 32, line_value: int = 128, line_length: int = 1):
        self.n = n
        self.size = size - line_length * 2
        self.line_value = line_value
        self.line_length = line_length

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
                self._compute_border_coordinate(
                    base + i * value, index, border_type
                )
                for index in range(init, fin)
            ]
        return result

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
        square_size = self.size + self.line_length * 2
        image = torch.zeros(
            self.n * square_size,
            self.n * square_size
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
                    self._compute_border("y", top_y, top_x, bottom_x) +
                    self._compute_border("y", bottom_y - 1, top_x, bottom_x, -1) +
                    self._compute_border("x", top_x, top_y, bottom_y) +
                    self._compute_border("x", bottom_x - 1, top_y, bottom_y, -1)
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
        state = state.reshape(self.n, self.n)
        result = torch.zeros(self.n * self.n, self.n, self.n, dtype=torch.bool)

        for i in range(self.n):
            for j in range(self.n):
                result[i * self.n + j] = state.clone()

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

                result[
                    i * self.n + j, top_x:bottom_x, top_y:bottom_y
                ] = ~state[top_x:bottom_x, top_y:bottom_y]

        return result.reshape(self.n * self.n, self.n * self.n)
