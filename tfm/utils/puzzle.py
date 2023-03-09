from typing import List, Union, Tuple

import numpy as np

from tfm.constants import ORDERED_ORDER
from tfm.utils.data import to_numpy


def has_correct_order(order: Union[np.asarray, List[int], Tuple[int, ...]]) -> bool:
    """Returns true if the `order` has values from 0 to 9 without any number
    repeated.

    Parameters
    ----------
    order: List[int]
        Order to be checked.

    Returns
    -------
    bool
        True if the order is correct false otherwise."""
    return all(
        map(lambda items: items[0] == items[1], zip(sorted(order), ORDERED_ORDER))
    ) and len(order) == len(ORDERED_ORDER)


def index_to_coordinates(index: int) -> Tuple[int, int]:
    """Transforms and index of the 8-puzzle to a coordinate. In instance, if we
    have the index 4, it corresponds to the row 1 column 1.

    Parameters
    ----------
    index: int
        Current index on the 8-puzzle order.

    Returns
    -------
    Row: int
        Row on the 8-puzzle.
    Column: int
        Column on the 8-puzzle."""
    return index // 3, index % 3


def manhattan_distance(index: int, original_index: int):
    """Computes the Manhattan index of an 8-puzzle given the current index of a
    digit and the position on the sorted puzzle.

    Parameters
    ----------
    index: int
        Current index of the digit.
    original_index: int
        Index that should have the digit on the sorted puzzle.

    Returns
    -------
    distance: int
        Manhattan distance"""
    row, column = index_to_coordinates(index)
    original_row, original_column = index_to_coordinates(original_index)

    return np.sum(np.abs([row - original_row, column - original_column]))


def is_solvable(
    sequence: Union[np.ndarray, List[int], Tuple[int, ...]],
    order: Union[np.ndarray, List[int], Tuple[int, ...]],
) -> bool:
    """Check if an 8-puzzle is solvable for a given order.

    Parameters
    ----------
    sequence: Union[np.ndarray, List[int], Tuple[int, ...]]
        Sequence of the problem or initial state.
    order: Union[np.ndarray, List[int], Tuple[int, ...]]
        Order of the problem or goal state.

    Returns
    -------
    bool
        True if the issue is solvable false otherwise."""
    sequence = to_numpy(sequence)
    order = to_numpy(order)
    digits_to_check = len(order) - 1
    start_digits_to_check = 1

    total = 0
    for i in range(digits_to_check):
        digit = order[i]
        for j in range(start_digits_to_check, digits_to_check):
            digit_to_check = order[j]

            digit_index = np.where(sequence == digit)[0][0]
            digit_to_check_index = np.where(sequence == digit_to_check)[0][0]

            total += int((i < j) != (digit_index < digit_to_check_index))
        start_digits_to_check += 1

    zero_index = np.where(sequence == 0)[0][0]
    original_zero_index = np.where(order == 0)[0][0]

    return total % 2 == manhattan_distance(zero_index, original_zero_index)
