#!/usr/bin/env python3

from . import context
import unittest
import poppy.simulators

import numpy as np


class SimulatorsTest(unittest.TestCase):

    def setUp(self):
        self.update_matrix_1 = np.array([1, 2, 3, 4])
        self.initial_conditions_1 = np.array([1, 2, 3, 4])
        self.rate_vector_1 = np.array([1, 2, 3, 4])
        self.t_max_1 = 111

        self.update_matrix_2 = np.array([1, 2, 3, 4])
        self.initial_conditions_2 = np.array([1, 2, 3, 4])
        self.rate_vector_2 = np.array([1, 2, 3, 4])
        self.t_max_2 = 111

    def test_SSA_1(self):
        output = poppy.simulators.SSA(self.update_matrix_1,
                                      self.initial_conditions_1,
                                      self.rate_vector_1,
                                      self.t_max_1)
        self.assertEqual(np.array_equiv(output, np.array([4, 5, 6])), True)

    def test_SSA_2(self):
        output = poppy.simulators.SSA(self.update_matrix_2,
                                      self.initial_conditions_2,
                                      self.rate_vector_2,
                                      self.t_max_2)
        self.assertEqual(np.array_equiv(output, np.array([7, 8, 9])), True)

    def test_Gillespie_1(self):
        output = poppy.simulators.Gillespie(self.update_matrix_1,
                                            self.initial_conditions_1,
                                            self.rate_vector_1,
                                            self.t_max_1)
        self.assertEqual(np.array_equiv(output, np.array([4, 5, 6])), True)

    def tearDown(self):
        # Here we can place repetitive code that should be performed _after_ every test is executed.
        pass
