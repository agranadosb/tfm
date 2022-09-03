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
