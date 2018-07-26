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
        times_and_states_trajectories = ssa.SSA(self.update_matrix_1,
                                                self.initial_conditions_1,
                                                self.rate_functions_1,
                                                self.t_max_1)

        exp_trajectory_times_and_states = np.array([[9.47936510e+01, 6., 0., 4.],
                                                    [9.68271008e+01, 7., 0., 3.],
                                                    [1.51137697e+02, 8., 0., 2.]])

        exp_trajectory_times_and_states_shape = (30, 4)

        self.assertEqual(times_and_states_trajectories.shape,
                         exp_trajectory_times_and_states_shape)
        self.assertTrue(np.allclose(times_and_states_trajectories[-3:, :],
                                    exp_trajectory_times_and_states))

    def test_next_reaction_method(self):

        secondary_parameters = {'affects': self.nrm_affects_1, 'depends_on': self.nrm_depends_on_1}

        times_and_states_trajectories = nrm.next_reaction_method(self.update_matrix_1,
                                                                 self.initial_conditions_1,
                                                                 self.rate_functions_1,
                                                                 self.t_max_1,
                                                                 **secondary_parameters
                                                                 )

        exp_trajectory_times_and_states = np.array([[82.76564785, 0., 0., 10.],
                                                    [85.69228982, 1., 0., 9.],
                                                    [91.22928178, 2., 0., 8.]])

        exp_trajectory_times_and_states_shape = (27, 4)

        self.assertEqual(times_and_states_trajectories.shape,
                         exp_trajectory_times_and_states_shape)
        self.assertTrue(np.allclose(times_and_states_trajectories[-3:, :],
                                    exp_trajectory_times_and_states))

    def tearDown(self):
        # Reset the numpy seed to a random value.
        np.random.seed()
