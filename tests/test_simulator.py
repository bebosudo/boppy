#!/usr/bin/env python3

from . import context
import unittest
import boppy.simulators.ssa as ssa
import boppy.simulators.next_reaction_method as nrm

import numpy as np


class SimulatorsTest(unittest.TestCase):

    def setUp(self):
        self.update_matrix_1 = np.array([[-1, 1, 0], [0, -1, 1], [1, 0, -1]])
        self.initial_conditions_1 = np.array([8, 2, 0])
        self.rate_functions_1 = lambda var: np.array([1 * var[1] * var[0] / 10,
                                                      0.05 * var[1],
                                                      0.01 * var[2]])
        self.t_max_1 = 100
        self.nrm_affects_1 = [[0, 1], [1, 2], [0, 2]]
        self.nrm_depends_on_1 = [[0, 1], [1], [2]]

        self.update_matrix_2 = np.array([1, 2, 3, 4])
        self.initial_conditions_2 = np.array([1, 2, 3, 4])
        self.rate_vector_2 = np.array([1, 2, 3, 4])
        self.t_max_2 = 111
        np.random.seed(42)

    def test_SSA_1(self):
        trajectory_states, trajectory_times = ssa.SSA(self.update_matrix_1,
                                                      self.initial_conditions_1,
                                                      self.rate_functions_1,
                                                      self.t_max_1)

        exp_trajectory_states = np.array([[8, 2, 0], [7, 3, 0], [6, 4, 0], [5, 5, 0], [4, 6, 0],
                                          [3, 7, 0], [2, 8, 0], [2, 7, 1], [1, 8, 1], [0, 9, 1],
                                          [0, 8, 2], [0, 7, 3], [0, 6, 4], [0, 5, 5], [0, 4, 6],
                                          [0, 3, 7], [0, 2, 8], [0, 1, 9], [1, 1, 8], [0, 2, 8],
                                          [1, 2, 7], [0, 3, 7], [0, 2, 8], [0, 1, 9], [0, 0, 10],
                                          [1, 0, 9], [2, 0, 8], [3, 0, 7], [4, 0, 6], [5, 0, 5]])

        exp_trajectory_times = np.array([0.00000000, 0.02972734, 0.25772599,
                                         0.97210810, 1.02434502, 1.15218482,
                                         1.16465382, 1.93925373, 2.90267191,
                                         3.43550703, 6.11681476, 10.80519096,
                                         13.44719320, 14.15842870, 16.37503388,
                                         28.17241541, 36.21056176, 36.50201573,
                                         38.02114339, 48.13077721, 52.68914859,
                                         54.58844114, 55.02047659, 57.30742185,
                                         61.97673176, 78.85418798, 81.68409764,
                                         83.07300677, 84.23497812, 111.3902763])

        self.assertEqual(trajectory_states.shape, exp_trajectory_states.shape)
        self.assertEqual(np.allclose(trajectory_states, exp_trajectory_states), True)
        self.assertEqual(trajectory_states.shape, exp_trajectory_states.shape)
        self.assertEqual(np.allclose(trajectory_times, exp_trajectory_times), True)

    def test_next_reaction_method(self):

        secondary_parameters = {'affects': self.nrm_affects_1, 'depends_on': self.nrm_depends_on_1}

        trajectory_states, trajectory_times = nrm.next_reaction_method(self.update_matrix_1,
                                                                       self.initial_conditions_1,
                                                                       self.rate_functions_1,
                                                                       self.t_max_1,
                                                                       **secondary_parameters
                                                                       )

        exp_trajectory_states = np.array([[8, 2, 0], [8, 1, 1], [7, 2, 1], [6, 3, 1], [5, 4, 1],
                                          [4, 5, 1], [3, 6, 1], [3, 5, 2], [2, 6, 2], [1, 7, 2],
                                          [0, 8, 2], [0, 7, 3], [0, 6, 4], [0, 5, 5], [0, 4, 6],
                                          [0, 3, 7], [0, 2, 8], [1, 2, 7], [0, 3, 7], [0, 2, 8],
                                          [0, 1, 9], [1, 1, 8], [0, 2, 8], [0, 1, 9], [0, 0, 10],
                                          [1, 0, 9], [2, 0, 8]])

        exp_trajectory_times = np.array([0.00000000, 0.50541675, 0.72215369,
                                         2.04924969, 3.63028953, 3.70212303,
                                         3.95660752, 4.13785332, 4.15049748,
                                         4.17595761, 4.43794469, 13.88975128,
                                         18.76035349, 24.41389058, 29.17361476,
                                         32.39772008, 37.99409988, 43.38133488,
                                         45.83765185, 48.83291652, 58.87425443,
                                         67.59444266, 70.01291698, 72.29467578,
                                         82.76564785, 85.69228982, 91.22928178])

        self.assertEqual(trajectory_states.shape, exp_trajectory_states.shape)
        self.assertEqual(np.allclose(trajectory_states,
                                     exp_trajectory_states), True)

        self.assertEqual(trajectory_times.shape, exp_trajectory_times.shape)
        self.assertEqual(np.allclose(trajectory_times,
                                     exp_trajectory_times), True)

    # def test_Gillespie_1(self):
    #     output = boppy.simulators.Gillespie(self.update_matrix_1,
    #                                         self.initial_conditions_1,
    #                                         self.rate_vector_1,
    #                                         self.t_max_1)
    #     self.assertEqual(np.allclose(output, np.array([4, 5, 6])), True)

    def tearDown(self):
        np.random.seed()
