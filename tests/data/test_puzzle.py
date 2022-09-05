from unittest import TestCase

import torch

from tfm.data.puzzle import Puzzle8MnistGenerator


class TestGet(TestCase):
    def setUp(self) -> None:
        self.generator = Puzzle8MnistGenerator()

    def test_get_correct_types(self):
        ordered = False

        puzzle, order = self.generator.get(ordered=ordered)

        self.assertTrue(isinstance(puzzle, torch.Tensor))
        self.assertTrue(isinstance(order, list))
        self.assertTrue(isinstance(order[0], int))

    def test_get_ordered_puzzle(self):
        ordered = True

        puzzle, order = self.generator.get(ordered=ordered)

        self.assertListEqual(list(sorted(order)), list(range(9)))

    def test_get_unordered_puzzle(self):
        ordered = False

        puzzle, order = self.generator.get(ordered=ordered)

        self.assertEqual(sum(order), sum(list(range(9))))

    def test_get_ordered_sequence_puzzle(self):
        sequence = [0, 1, 4, 2, 3, 5, 6, 7, 8]
        ordered = True

        puzzle, order = self.generator.get(ordered=ordered, sequence=sequence)

        self.assertListEqual(list(sorted(order)), list(range(9)))

    def test_get_unordered_sequence_puzzle(self):
        sequence = [0, 1, 4, 2, 3, 5, 6, 7, 8]
        ordered = False

        puzzle, order = self.generator.get(ordered=ordered, sequence=sequence)

        self.assertListEqual(order, sequence)
