from unittest import TestCase

import numpy as np

from tfm.utils.data import to_numpy


class TestUtilsData(TestCase):
    def test_to_numpy_array_from_list(self):
        iterable = list(range(9))

        result = to_numpy(iterable)

        self.assertTrue(isinstance(result, np.ndarray))

    def test_to_numpy_array_from_tuple(self):
        iterable = tuple(range(9))

        result = to_numpy(iterable)

        self.assertTrue(isinstance(result, np.ndarray))

    def test_to_numpy_array_from_numpy(self):
        iterable = np.asarray(list(range(9)))

        result = to_numpy(iterable)

        self.assertTrue(isinstance(result, np.ndarray))
