from unittest import TestCase

import numpy as np
import torch

from tfm.data.puzzle import Puzzle8MnistGenerator
from tfm.utils.puzzle import is_solvable


class TestPuzzle8MnistGenerator(TestCase):
    def setUp(self) -> None:
        self.generator = Puzzle8MnistGenerator()

    def test___init__custom_different_digits(self):
        different_digits = 1
        puzzle = Puzzle8MnistGenerator(different_digits=different_digits)

        for value in puzzle.indices.values():
            self.assertEqual(different_digits, len(value))

    def test_get_correct_types(self):
        ordered = False

        puzzle, order = self.generator.get(ordered=ordered)

        self.assertTrue(isinstance(puzzle, torch.Tensor))
        self.assertTrue(isinstance(order, np.ndarray))
        self.assertTrue(isinstance(order[0], np.int16))

    def test_get_ordered_puzzle(self):
        ordered = True

        puzzle, order = self.generator.get(ordered=ordered)

        self.assertListEqual(list(sorted(order)), list(range(9)))

    def test_get_unordered_puzzle(self):
        ordered = False

        puzzle, order = self.generator.get(ordered=ordered)

        self.assertEqual(sum(order), sum(list(range(9))))

    def test_get_ordered_list_sequence_puzzle(self):
        sequence = [0, 1, 4, 2, 3, 5, 6, 7, 8]
        ordered = True

        puzzle, order = self.generator.get(ordered=ordered, sequence=sequence)

        self.assertListEqual(list(sorted(order)), list(range(9)))

    def test_get_unordered_list_sequence_puzzle(self):
        sequence = [0, 1, 4, 2, 3, 5, 6, 7, 8]
        numpy_sequence = np.asarray(sequence, dtype=np.int16)
        ordered = False

        puzzle, order = self.generator.get(ordered=ordered, sequence=sequence)

        self.assertTrue(np.array_equal(numpy_sequence, order))

    def test_get_ordered_sequence_puzzle(self):
        sequence = np.asarray([0, 1, 4, 2, 3, 5, 6, 7, 8], dtype=np.int16)
        ordered = True

        puzzle, order = self.generator.get(ordered=ordered, sequence=sequence)

        self.assertListEqual(list(sorted(order)), list(range(9)))

    def test_get_unordered_sequence_puzzle(self):
        sequence = np.asarray([0, 1, 4, 2, 3, 5, 6, 7, 8], dtype=np.int16)
        ordered = False

        puzzle, order = self.generator.get(ordered=ordered, sequence=sequence)

        self.assertTrue(np.array_equal(np.asarray(sequence), order))

    def test_get_custom_order(self):
        correct_order = [1, 2, 3, 8, 0, 4, 7, 6, 5]
        ordered = True
        generator = Puzzle8MnistGenerator(order=correct_order)

        puzzle, order = generator.get(ordered=ordered)

        self.assertTrue(np.array_equal(np.asarray(correct_order), order))

    def test_get_incorrect_custom_order(self):
        order = [1, 2, 3, 8, 0, 4, 7, 6, 4]

        with self.assertRaises(ValueError):
            Puzzle8MnistGenerator(order=order)

    def test__random_movements(self):
        order = [1, 2, 3, 8, 0, 4, 7, 6, 5]
        puzzle = Puzzle8MnistGenerator(order=order)

        np.random.seed(7)
        for _ in range(50):
            sequence_result, movements_result = puzzle._random_movements()

            self.assertEqual(sum(sequence_result), sum(list(range(9))))
            self.assertIn(4 + sum(movements_result), list(range(9)))
            self.assertTrue(is_solvable(sequence_result, order))
