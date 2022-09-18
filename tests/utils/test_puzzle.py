from unittest import TestCase

from tfm.utils.puzzle import has_correct_order


class TestUtilsPuzzle(TestCase):
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
        pass
