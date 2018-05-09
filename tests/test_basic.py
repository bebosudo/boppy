#!/usr/bin/env python3

from .context import poppy
import unittest


class ExampleTest(unittest.TestCase):

    def setUp(self):
        # Here we can place repetitive code that should be performed _before_ every test is executed.
        pass

    def test_square_of_number(self):
        self.assertTrue(poppy.square(3) == 9)

    def tearDown(self):
        # Here we can place repetitive code that should be performed _after_ every test is executed.
        pass


if __name__ == '__main__':
    unittest.main()
