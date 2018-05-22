#!/usr/bin/env python3

import numpy as np

import unittest
from .context import poppy


class SimulatorTest(unittest.TestCase):

    def setUp(self):
        self.update_matrix = np.array([1, 2, 3, 4])
        self.initial_conditions = np.array([1, 2, 3, 4])
        self.rate_vector = np.array([1, 2, 3, 4])
        self.t_max = 111

    def test_non_existing_filename(self):
        output = poppy.simulators.SSA(self.input_array,
                                      self.initial_conditions,
                                      self.rate_vector,
                                      self.t_max)
        self.assertEqual(output == [4, 5, 6])

    def tearDown(self):
        # Here we can place repetitive code that should be performed _after_ every test is executed.
        pass


if __name__ == "__main__":
    unittest.main()
