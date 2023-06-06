from unittest import TestCase
from unittest.mock import patch

import torch

from tfm.data.puzzle import Puzzle8MnistGenerator


class TestPuzzle8MnistGenerator(TestCase):
    def setUp(self) -> None:
        self.generator = Puzzle8MnistGenerator()

    def test__init__incorrect_order(self):
        with self.assertRaises(ValueError):
            Puzzle8MnistGenerator([0] * 10)

    def test__init__digits_indices(self):
        for correct_index, (index, indices) in enumerate(
            self.generator.digits_indices.items()
        ):
            self.assertEqual(correct_index, index)
            self.assertEqual(len(indices), 10)

    def test__is_beyond_bounds_beyond_bounds(self):
        is_beyond_bounds = self.generator._is_beyond_bounds(1, -3)

        self.assertTrue(is_beyond_bounds)

    def test__is_beyond_bounds_within_bounds(self):
        is_beyond_bounds = self.generator._is_beyond_bounds(1, 1)

        self.assertFalse(is_beyond_bounds)

    def test__is_beyond_bounds_within_bounds_incorrect_move(self):
        is_beyond_bounds = self.generator._is_beyond_bounds(2, 1)

        self.assertFalse(is_beyond_bounds)

    def test__is_incorrect_left_incorrect(self):
        is_incorrect = self.generator._is_incorrect_left(3, -1)

        self.assertTrue(is_incorrect)

    def test__is_incorrect_left_correct(self):
        is_incorrect = self.generator._is_incorrect_left(4, -1)

        self.assertFalse(is_incorrect)

    def test__is_incorrect_right_incorrect(self):
        is_incorrect = self.generator._is_incorrect_right(2, 1)

        self.assertTrue(is_incorrect)

    def test__is_incorrect_right_correct(self):
        is_incorrect = self.generator._is_incorrect_right(4, 1)

        self.assertFalse(is_incorrect)

    def test_is_possible_incorrect_left(self):
        is_possible = self.generator.is_possible(3, -1)

        self.assertFalse(is_possible)

    def test_is_possible_incorrect_right(self):
        is_possible = self.generator.is_possible(2, 1)

        self.assertFalse(is_possible)

    def test_is_possible_beyond_bounds(self):
        is_possible = self.generator.is_possible(1, -3)

        self.assertFalse(is_possible)

    def test_is_possible_correct(self):
        is_possible = self.generator.is_possible(1, 1)

        self.assertTrue(is_possible)

    def test_zero_index(self):
        zero_index = self.generator.zero_index(torch.arange(9))

        self.assertEqual(zero_index, 0)

    def test_move(self):
        order = torch.tensor([1, 2, 3, 8, 0, 4, 7, 6, 5])
        correct_new_order = torch.tensor([1, 2, 3, 8, 4, 0, 7, 6, 5])
        new_order = self.generator.move(order, 1)

        self.assertTrue(torch.equal(new_order, correct_new_order))

    def test_move_incorrect(self):
        order = torch.tensor([1, 0, 3, 8, 2, 4, 7, 6, 5])

        with self.assertRaises(ValueError):
            self.generator.move(order, -3)

    def test__move(self):
        order = torch.tensor([1, 2, 3, 8, 0, 4, 7, 6, 5])
        correct_new_order = torch.tensor([1, 2, 3, 8, 4, 0, 7, 6, 5])
        new_order, new_zero_index = self.generator._move(order, 4, 1)

        self.assertTrue(torch.equal(new_order, correct_new_order))
        self.assertEqual(new_zero_index, 5)

    def test__all_moves_all_correct(self):
        order = torch.tensor([1, 2, 3, 8, 0, 4, 7, 6, 5])
        zero_index = 4
        correct_movements = [0, 1, 2, 3]
        new_orders, new_zero_index, movements = self.generator._all_moves(
            order, zero_index
        )

        self.assertIn(new_zero_index, [3, 5, 7, 1])
        for movement, correct_movement in zip(sorted(movements), correct_movements):
            self.assertEqual(movement, correct_movement)

    def test__all_moves_incorrect(self):
        order = torch.tensor([1, 0, 3, 8, 2, 4, 7, 6, 5])
        zero_index = 1
        correct_movements = [0, 1, 2, 4]
        new_orders, new_zero_index, movements = self.generator._all_moves(
            order, zero_index
        )

        self.assertIn(new_zero_index, [0, 2, 4])
        for movement, correct_movement in zip(sorted(movements), correct_movements):
            self.assertEqual(movement, correct_movement)

    @patch("tfm.data.puzzle.random.choice", return_value=1)
    def test_random_move_possible(self, *_):
        order = torch.tensor([1, 2, 3, 8, 0, 4, 7, 6, 5])
        zero_index = 4
        correct_new_order = torch.tensor([1, 2, 3, 8, 4, 0, 7, 6, 5])
        correct_new_zero_index = 5

        new_order, new_zero_index, label = self.generator.random_move(order, zero_index)

        self.assertTrue(torch.equal(new_order, correct_new_order))
        self.assertEqual(new_zero_index, correct_new_zero_index)
        self.assertEqual(label, 0)

    @patch("tfm.data.puzzle.random.choice", return_value=-3)
    def test_random_move_not_possible(self, *_):
        order = torch.tensor([1, 0, 3, 8, 2, 4, 7, 6, 5])
        zero_index = 1
        correct_new_order = torch.tensor([1, 2, 3, 8, 0, 4, 7, 6, 5])
        correct_new_zero_index = 4

        new_order, new_zero_index, label = self.generator.random_move(order, zero_index)

        self.assertTrue(torch.equal(new_order, correct_new_order))
        self.assertEqual(new_zero_index, correct_new_zero_index)
        self.assertEqual(label, 2)
