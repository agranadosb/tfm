from typing import List, Tuple, Union

import numpy as np


def to_numpy(iterable: Union[np.ndarray, List[int], Tuple[int, ...]]) -> np.ndarray:
    """Transforms an iterable to numpy array.

    Parameters
    ----------
    iterable: Union[np.ndarray, List[int], Tuple[int, ...]]
        Iterable to transform to numpy array.

    Returns
    -------
    np.ndarray
        Numpy array."""
    if not isinstance(iterable, np.ndarray):
        return np.asarray(iterable)
    return iterable
