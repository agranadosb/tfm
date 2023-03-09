from unittest import TestCase

from tfm.utils.puzzle import (
    has_correct_order,
    is_solvable,
    index_to_coordinates,
    manhattan_distance,
)


class TestUtilsPuzzle(TestCase):
    def test_index_to_coordinates(self):
        index = 4
        correct_row = 1
        correct_column = 1

        row, column = index_to_coordinates(index)

        self.assertEqual(correct_row, row)
        self.assertEqual(correct_column, column)

    def test_manhattan_distance(self):
        index = 4
        original_index = 0
        correct_distance = 2

        distance = manhattan_distance(index, original_index)

        self.assertEqual(correct_distance, distance)

    def test_has_correct_order_correct(self):
        order = [0, 1, 2, 6, 7, 8, 3, 4, 5]

        result = has_correct_order(order)

        self.assertTrue(result)

    def test_has_correct_order_correct_order_incorrect_length(self):
        order = [0, 1, 2]

        result = has_correct_order(order)

        self.assertFalse(result)

    def test_has_correct_order_incorrect_order_incorrect_length(self):
        order = [0, 1, 0]

        result = has_correct_order(order)

        self.assertFalse(result)

    def test_has_correct_order_incorrect_order_correct_length(self):
        order = [0, 1, 2, 6, 7, 8, 3, 4, 0]

        result = has_correct_order(order)

        self.assertFalse(result)

    def test_is_solvable_solvable_puzzle(self):
        order = [1, 2, 3, 8, 0, 4, 7, 6, 5]
        sequence = [2, 8, 3, 1, 6, 4, 7, 0, 5]

        result = is_solvable(sequence, order)

        self.assertTrue(result)

    def test_is_solvable_solvable_2_puzzle(self):
        order = [1, 2, 3, 0, 8, 4, 7, 6, 5]
        sequence = [2, 8, 3, 1, 6, 4, 7, 0, 5]

        result = is_solvable(sequence, order)

        self.assertFalse(result)

    def test_is_solvable_same(self):
        order = [1, 2, 3, 8, 0, 4, 7, 6, 5]
        sequence = [1, 2, 3, 8, 0, 4, 7, 6, 5]

        result = is_solvable(sequence, order)

        self.assertTrue(result)

    def test_is_solvable_solvable_puzzle_2(self):
        order = [1, 2, 3, 8, 0, 4, 7, 6, 5]
        sequence = [1, 2, 0, 7, 4, 3, 6, 8, 5]

        result = is_solvable(sequence, order)

        self.assertFalse(result)
