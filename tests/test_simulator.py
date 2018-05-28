#!/usr/bin/env python3

from . import context
import unittest
import poppy.simulators

import numpy as np


class SimulatorsTest(unittest.TestCase):

    def setUp(self):
        self.update_matrix_1 = np.array([[-1, 1, 0], [0, -1, 1], [1, 0, -1]])
        self.initial_conditions_1 = np.array([8, 2, 0])
        self.rate_functions_1 = [lambda x_s, x_i, x_r: np.array(
            [1 * x_i * x_s / 10, 0.05 * x_i, 0.01 * x_r])]
        self.nrm_affects_1 = [[0, 1], [1, 2], [0, 2]]
        self.nrm_depends_on_1 = [[0, 1], [1], [2]]
        self.t_max_1 = 100

        self.update_matrix_2 = np.array([1, 2, 3, 4])
        self.initial_conditions_2 = np.array([1, 2, 3, 4])
        self.rate_vector_2 = np.array([1, 2, 3, 4])
        self.t_max_2 = 111
        np.random.seed(42)

    def test_SSA_1(self):
        output = poppy.simulators.SSA(self.update_matrix_1,
                                      self.initial_conditions_1,
                                      self.rate_functions_1,
                                      self.t_max_1)
        self.assertEqual(np.array_equiv(output, np.array([4, 5, 6])), True)

    def test_next_reaxtion_method(self):
        output = poppy.simulators.next_reaction_method(self.update_matrix_1,
                                                       self.initial_conditions_1,
                                                       self.function_rates_1,
                                                       self.t_max_1,
                                                       self.nrm_affects_1,
                                                       self.nrm_depends_on_1
                                                       )
        self.assertEqual(np.array_equiv(output, np.array([7, 8, 9])), True)

    # def test_Gillespie_1(self):
    #     output = poppy.simulators.Gillespie(self.update_matrix_1,
    #                                         self.initial_conditions_1,
    #                                         self.rate_vector_1,
    #                                         self.t_max_1)
    #     self.assertEqual(np.array_equiv(output, np.array([4, 5, 6])), True)

    def tearDown(self):
        np.random.seed()
