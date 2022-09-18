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
    order, order_indices = np.unique(order, return_index=True)
    sequence_indices = np.unique(sequence, return_index=True)[1]

    total = 0
    for digit in range(np.max(order) + 1):
        on_right = order_indices > order_indices[digit]
        on_left = sequence_indices < sequence_indices[digit]
        total += np.logical_and(on_right, on_left).sum()

    return total % 2 == 0
