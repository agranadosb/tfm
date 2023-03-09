from unittest import TestCase

import numpy as np

from tfm.constants import MOVEMENTS
from tfm.data.puzzle import Puzzle8MnistGenerator


class TestPuzzle8MnistGenerator(TestCase):
    def setUp(self) -> None:
        self.generator = Puzzle8MnistGenerator()

    def test___init__custom_different_digits(self):
        different_digits = 1
        puzzle = Puzzle8MnistGenerator(different_digits=different_digits)

        for value in puzzle.indices.values():
            self.assertEqual(different_digits, len(value))

    def test_get_incorrect_custom_order(self):
        order = [1, 2, 3, 8, 0, 4, 7, 6, 4]

        with self.assertRaises(ValueError):
            Puzzle8MnistGenerator(order=order)

    def test_random_permutation(self):
        order = [1, 2, 3, 8, 0, 4, 7, 6, 5]
        puzzle = Puzzle8MnistGenerator(order=order)
        current_order = np.asarray(order)
        zero_index = 4

        np.random.seed(7)
        for _ in range(50):
            movement, zero_index = puzzle.random_permutation(current_order, zero_index)

            self.assertEqual(current_order[zero_index], 0)
            self.assertIn(movement, MOVEMENTS)
            self.assertEqual(sum(current_order), sum(list(range(9))))
            sorted_order = current_order.copy()
            sorted_order: list = list(sorted(sorted_order.tolist()))  # noqa
            self.assertListEqual(sorted_order, list(range(9)))
