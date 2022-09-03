from typing import List, Tuple

import numpy as np
import torch


class Puzzle8MnistGenerator:
    """This class generates a random 8.puzzle based on mnist digits.

    Parameters
    ----------
    size: int
        Size of the image representing the puzzle."""

    def __init__(self, size: int):
        self.size = size

    def get(self, ordered: bool = False) -> Tuple[torch.Tensor, List[int]]:
        """Returns a random generated 8-puzzle.

        Parameters
        ----------
        ordered: bool = False
            If True the 8-puzzle will be ordered.

        Returns
        -------
        Tuple[np.ndarray, List[int]]
            8-Puzzle generated.
            Order of the digits on the mnist puzzle."""
