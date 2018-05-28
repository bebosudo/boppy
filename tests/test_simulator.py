#!/usr/bin/env python3

from . import context
import unittest
import poppy.simulators.next_reaction_method

import numpy as np


class SimulatorsTest(unittest.TestCase):

    def setUp(self):
        self.update_matrix_1 = np.array([[-1, 1, 0], [0, -1, 1], [1, 0, -1]])
        self.initial_conditions_1 = np.array([8, 2, 0])
        self.rate_functions_1 = lambda var: np.array(
            [1 * var[1] * var[0] / 10, 0.05 * var[1], 0.01 * var[2]])
        self.t_max_1 = 100
        self.nrm_affects_1 = [[0, 1], [1, 2], [0, 2]]
        self.nrm_depends_on_1 = [[0, 1], [1], [2]]

        self.update_matrix_2 = np.array([1, 2, 3, 4])
        self.initial_conditions_2 = np.array([1, 2, 3, 4])
        self.rate_vector_2 = np.array([1, 2, 3, 4])
        self.t_max_2 = 111
        np.random.seed(42)

    def test_SSA_1(self):
        trajectory_states, trajectory_times = poppy.simulators.ssa.SSA(self.update_matrix_1,
                                      self.initial_conditions_1,
                                      self.rate_functions_1,
                                      self.t_max_1)
	expected_trajectory_states = np.array([[8, 2, 0], [7, 3, 0], [6, 4, 0], [5, 5, 0], [4, 6, 0],
       					       [3, 7, 0], [2, 8, 0], [2, 7, 1], [1, 8, 1], [0, 9, 1],
       					       [0, 8, 2], [0, 7, 3], [0, 6, 4], [0, 5, 5], [0, 4, 6],
      					       [0, 3, 7], [0, 2, 8], [0, 1, 9], [1, 1, 8], [0, 2, 8],
         				       [1, 2, 7], [0, 3, 7], [0, 2, 8], [0, 1, 9], [0, 0, 10],
      					       [1, 0, 9], [2, 0, 8], [3, 0, 7], [4, 0, 6], [5, 0, 5]])
	
	expected_trajectory_times = np.array([0, 0.02972734787254852, 0.25772599178629474, 0.972108105845827, 
		 	    		      1.0243450293644674,  1.1521848263720367, 1.1646538244774989,
					      1.9392537382702066, 2.9026719118110473, 3.4355070384984283,
 					      6.116814768243328, 10.805190961374164, 13.447193203091764,
 					      14.158428709166536, 16.37503388342292, 28.172415419157556,
 					      36.21056176161607, 36.502015733482736, 38.02114339494848,
 					      48.13077721307886, 52.68914859937929, 54.58844114875421,
 					      55.02047659051431, 57.30742185665263, 61.976731768953684,
 					      78.8541879838408, 81.68409764611367, 83.073006777784,
 					      84.23497812963193, 111.39027639037134])
        self.assertEqual(np.array_equiv(trajectory_states, expected_trajectory_states, True)
        self.assertEqual(np.array_equiv(trajectory_times, expected_trajectory_times, True)

    def test_next_reaction_method(self):
        trajectory_states, trajectory_times = poppy.simulators.next_reaction_method.next_reaction_method(self.update_matrix_1,
                                                                                                         self.initial_conditions_1,
                                                                                                         self.rate_functions_1,
                                                                                                         self.t_max_1,
                                                                                                         self.nrm_affects_1,
                                                                                                         self.nrm_depends_on_1
                                                                                                         )
        expected_trajectory_states = np.array([[8, 2, 0], [8, 1, 1], [7, 2, 1], [6, 3, 1],
                                               [5, 4, 1], [4, 5, 1], [4, 4, 2],
                                               [3, 5, 2], [2, 6, 2], [2, 5, 3],
                                               [2, 4, 4], [2, 3, 5], [1, 4, 5],
                                               [0, 5, 5], [0, 4, 6], [0, 3, 7],
                                               [0, 2, 8], [0, 1, 9], [0, 0, 10]])
        expected_trajectory_times = np.array([0., 0.50541675, 0.72215369, 2.04913925,
                                              3.08132503, 4.50426089, 4.22471507,
                                              4.66393921, 4.89407833, 6.10604296,
                                              6.22825155, 7.14520634, 9.76701837,
                                              14.0287953, 15.24456935, 21.19422457,
                                              25.49303166, 33.88760137, 58.56049933])
        self.assertEqual(np.array_equiv(trajectory_states,
                                        expected_trajectory_states), True)
        self.assertEqual(np.allclose(trajectory_times,
                                     expected_trajectory_times), True)

    # def test_Gillespie_1(self):
    #     output = poppy.simulators.Gillespie(self.update_matrix_1,
    #                                         self.initial_conditions_1,
    #                                         self.rate_vector_1,
    #                                         self.t_max_1)
    #     self.assertEqual(np.array_equiv(output, np.array([4, 5, 6])), True)

    def tearDown(self):
        np.random.seed()
