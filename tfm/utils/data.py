from typing import List, Tuple, Union

from datetime import datetime
import numpy as np


def current_datetime() -> str:
    """Returns current datetime in format YYYYMMDD-HHMMSS.

    Returns
    -------
    str
        Current datetime."""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


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
