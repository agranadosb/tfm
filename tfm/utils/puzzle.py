from typing import List, Union, Tuple

import numpy as np

from tfm.constants import ORDERED_ORDER


def has_correct_order(
    order: Union[np.asarray, List[int], Tuple[int, ...]]
) -> bool:
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
