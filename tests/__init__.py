#!/usr/bin/env python3

import os
import sys
import unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import poppy


class ExampleTest(unittest.TestCase):

    def setUp(self):
        # Here we can place repetitive code that should be performed _before_ every test is executed.
        pass

    def test_square_of_number(self):
        self.assertTrue(poppy.square(3) == 9)

    def tearDown(self):
        # Here we can place repetitive code that should be performed _after_ every test is executed.
        pass


def run_tests():
    unittest.main()


if __name__ == '__main__':
    run_tests()
