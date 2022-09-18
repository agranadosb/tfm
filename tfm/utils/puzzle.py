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
):
    sequence = to_numpy(sequence)
    order = to_numpy(order)
    order_indices = np.unique(order, return_index=True)[1]
    sequence_indices = np.unique(sequence, return_index=True)[1]

    inversion_table = np.zeros((9, 9), dtype=bool)

    for digit in range(np.max(order)):
        on_right = order_indices > order_indices[digit]
        on_left = sequence_indices < sequence_indices[digit]
        inversion_table[
            digit,
            np.logical_and(on_right, on_left),
        ] = True

    return np.sum(inversion_table) % 2 == 1
