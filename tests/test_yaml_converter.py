from . import context
import unittest
import boppy.core
from boppy.utils.misc import BoppyInputError
import boppy.simulators
import boppy.file_parser

import os.path
from tempfile import TemporaryDirectory
import numpy as np
import sympy as sym


class YAMLTest(unittest.TestCase):
    """Test that the YAML converter meets the YAML standard."""

    def setUp(self):
        self.example_input_text = ('Species:\n  - x_s\n  - x_i\n  - x_r\n\nParameters:\n  k_s: '
                                   '0.01\n  k_i: 1\n  k_r: 0.05\n\nReactions:\n  - x_s + x_i => '
                                   'x_i + x_i\n  - x_i => x_r\n  - x_r => x_s\n\nRate functions:\n'
                                   '  - k_i * x_i * x_s / N\n  - k_r * x_i\n  - k_s * x_r\n\n'
                                   'Initial conditions:\n  x_s: 80\n  x_i: 20\n  x_r: 0\n\nSystem '
                                   'size:\n  N: 100\n\nSimulation: SSA\n\nObservables:\n  - tot = '
                                   'x + y\n\nProperties:\n  x_s: 43\n')
        self.expected_converted_dictionary = {'Species': ['x_s', 'x_i', 'x_r'],
                                              'Parameters': {'k_i': 1, 'k_r': 0.05, 'k_s': 0.01},
                                              'Reactions': ['x_s + x_i => x_i + x_i',
                                                            'x_i => x_r',
                                                            'x_r => x_s'],
                                              'Rate functions': ['k_i * x_i * x_s / N',
                                                                 'k_r * x_i', 'k_s * x_r'],
                                              'Initial conditions': {'x_i': 20, 'x_r': 0, 'x_s': 80},
                                              'Properties': {'x_s': 43},
                                              'Observables': ['tot = x + y'],
                                              'Simulation': 'SSA',
                                              'System size': {'N': 100}
                                              }

    def test_interpret_file_from_yaml_to_dict(self):
        self.assertTrue(self.expected_converted_dictionary ==
                        boppy.file_parser.yaml_string_to_dict_converter(self.example_input_text))

    def test_interpret_empty_file_from_yaml_to_dict(self):
        empty_input_file = ""
        self.assertIsNone(boppy.file_parser.yaml_string_to_dict_converter(empty_input_file))

    def test_filename_to_dict(self):
        with TemporaryDirectory() as tmpdirname:
            temp_filename = os.path.join(tmpdirname, "test_input.txt")
            with open(temp_filename, "w") as temp_fd:
                temp_fd.write(self.example_input_text)

            self.assertTrue(self.expected_converted_dictionary ==
                            boppy.file_parser.filename_to_dict_converter(temp_filename))

    def test_empty_filename(self):
        with TemporaryDirectory() as tmpdirname:
            temp_filename = os.path.join(tmpdirname, "test_input.txt")
            with open(temp_filename, "w") as temp_fd:
                temp_fd.write("")
            self.assertIsNone(boppy.file_parser.filename_to_dict_converter(temp_filename))

            with open(temp_filename, "w") as temp_fd:
                temp_fd.write("\n")
            self.assertIsNone(boppy.file_parser.filename_to_dict_converter(temp_filename))

    def test_non_existing_filename(self):
        with TemporaryDirectory() as tmpdirname:
            temp_filename = os.path.join(tmpdirname, "test_input.txt")
            with self.assertRaises(FileNotFoundError):
                boppy.file_parser.filename_to_dict_converter(temp_filename)

    def tearDown(self):
        # Here we can place repetitive code that should be performed _after_ every test is executed.
        pass


class BoppyCoreComponentsTest(unittest.TestCase):
    """Test that the parser is able to correctly convert input to the correct components."""

    def setUp(self):
        self.raw_input = {'Initial conditions': {'x_s': 80, 'x_i': 20, 'x_r': 0},
                          'Observables': ['tot = x + y'],
                          'Parameters': {'k_i': 1, 'k_r': 0.05, 'k_s': 0.01},
                          'Properties': {'x_s': 43},
                          'Rate functions': ['k_i * x_i * x_s / N', 'k_r * x_i', 'k_s * x_r'],
                          'Reactions': ['x_s + x_i => x_i + x_i', 'x_i => x_r', 'x_r => x_s'],
                          'Simulation': 'SSA',
                          'Maximum simulation time': 100,
                          'Species': ['x_s', 'x_i', 'x_r'],
                          'System size': {'N': 100}
                          }

        # Data is different from the raw version above.
        self.input_data = {"Species": boppy.core.VariableCollection(["x_s", "x_i", "x_r"]),
                           "Parameters": boppy.core.ParameterCollection({'k_i': 1,
                                                                         'k_r': 0.05,
                                                                         'k_s': 0.01,
                                                                         'N': 100}),
                           "Rate functions": ["4 - 6",
                                              "k_i * x_i * x_s /    N  ",
                                              "-3* 2 * k_i * x_i * x_s / max(x_s, x_i) + 2 * x_i - x_i + x_r"],
                           "Reactions": ["x_s => x_i",
                                         "x_s + x_i => x_r",
                                         "3x_s + x_i => x_r"],
                           "Initial conditions": np.array([80, 20, 0]),
                           }
        self.expected_update_vector = [np.array([-1, 1, 0]),
                                       np.array([-1, -1, 1]),
                                       np.array([-3, -1, 1])]

        x_s, x_i, x_r = sym.symbols("x_s x_i x_r")
        self.expected_rate_functions = [sym.Integer(-2),
                                        x_s * x_i / 100,
                                        -6.0 * x_s * x_i / sym.Max(x_i, x_s) + 1.0 * x_i + x_r]

        self.rate_func_coll = boppy.core.RateFunctionCollection(self.input_data["Rate functions"],
                                                                self.input_data["Species"],
                                                                self.input_data["Parameters"])

        np.random.seed(42)

    def test_convert_all_reactions(self):
        for i, _ in enumerate(self.input_data["Reactions"]):
            reaction_obj = boppy.core.Reaction(self.input_data["Reactions"][i],
                                               self.input_data["Species"])

            self.assertEqual(np.array_equiv(reaction_obj.update_vector,
                                            self.expected_update_vector[i]), True)

    def test_missing_value_raises_exc(self):
        with self.assertRaisesRegex(BoppyInputError, "Unable to find reagent '.*' inside the list of variables"):
            boppy.core.Reaction("R1 + R2 => 2 P1", self.input_data["Species"])

    def test_reaction_collection(self):
        reaction_collection = boppy.core.ReactionCollection(self.input_data["Reactions"],
                                                            self.input_data["Species"])

        for reac, upd_vec in zip(reaction_collection, self.expected_update_vector):
            self.assertEqual(np.array_equiv(reac.update_vector, upd_vec), True)

    def test_all_rate_functions(self):
        for i, _ in enumerate(self.input_data["Rate functions"]):
            rate_func = boppy.core.RateFunction(self.input_data["Rate functions"][i],
                                                self.input_data["Species"],
                                                self.input_data["Parameters"])
            self.assertEqual(rate_func.sym_function, self.expected_rate_functions[i])

    def test_rate_functions_collection_symbolic_arg(self):
        for i, rate_func in enumerate(self.rate_func_coll):
            self.assertEqual(rate_func.sym_function, self.expected_rate_functions[i])

    def test_rate_functions_collection_compute_in_points(self):
        self.assertEqual(True, np.allclose(self.rate_func_coll(self.input_data["Initial conditions"]),
                                           [-2., 16., -100.]))

    def test_rate_functions_collection_compute_dim_mismatch_exc(self):
        with self.assertRaisesRegex(BoppyInputError, "Array shapes mismatch: input vector \d, rate functions \d."):
            self.rate_func_coll(np.array([1, 2]))

    def test_application_controller_shape_rate_func_diff_parameters(self):
        with self.assertRaisesRegex(BoppyInputError, "The number of Parameters \(\d\) is different "
                                    "from the number of Rate functions \(\d\)"):
            self.raw_input["Rate functions"] = self.raw_input["Rate functions"][:-1]
            boppy.core.MainController(self.raw_input)

    def test_application_controller_unknown_algorithm(self):
        with self.assertRaisesRegex(BoppyInputError, "The algorithm to use in the simulation must "
                                    "be a string in '.*',?\."):
            self.raw_input["Simulation"] = "asdf"
            boppy.core.MainController(self.raw_input)

    def test_application_controller_multiple_algorithms_requested(self):
        with self.assertRaisesRegex(BoppyInputError, "The algorithm to use in the simulation must "
                                    "be a single string\."):
            self.raw_input["Simulation"] = None
            boppy.core.MainController(self.raw_input)

    def test_application_controller_multiple_dimension_sizes(self):
        with self.assertRaisesRegex(BoppyInputError, "The size of the system must be a single "
                                    "parameter\. Found: .*\."):
            self.raw_input["System size"] = {"N": 123, "M": 456}
            boppy.core.MainController(self.raw_input)

    def test_application_controller_missing_initial_condition(self):
        with self.assertRaisesRegex(BoppyInputError, "The maximum simulation time parameter \(t_max\) "
                                    "must be a number\. Found: .*\."):
            self.raw_input["Maximum simulation time"] = [123, 456]
            boppy.core.MainController(self.raw_input)

    def test_application_controller_correct_handling(self):
        controller = boppy.core.MainController(self.raw_input)
        x_s, x_i, x_r, N, k_i, k_s, k_r = sym.symbols("x_s x_i x_r N k_i k_s k_r")

        set_of_variables = {x_s, x_i, x_r}
        for _, var in controller._variables.items():
            self.assertIn(var.symbol, set_of_variables)

        dict_of_parameters = {k_i: 1, k_r: 0.05, k_s: 0.01, N: 100}
        for _, var in controller._parameters.items():
            self.assertIn(var.symbol, dict_of_parameters)
            self.assertEqual(var.value, dict_of_parameters[var.symbol])

        expected_raw_rate_functions = [x_i * x_s / 100, 0.05 * x_i, 0.01 * x_r]
        for idx, rfun in enumerate(controller._rate_functions):
            self.assertEqual(expected_raw_rate_functions[idx], rfun.sym_function)

        expected_raw_reactions_upd_vec = [np.array([-1.,  1.,  0.]),
                                          np.array([0., -1.,  1.]),
                                          np.array([1.,  0., -1.])]
        for idx, reac in enumerate(controller._reactions):
            self.assertEqual(True, np.allclose(expected_raw_reactions_upd_vec[idx],
                                               reac.update_vector))

        self.assertEqual(controller._system_size.value, 100)
        self.assertEqual(controller._t_max, 100)

        self.assertEqual(True, np.allclose(controller._initial_conditions, np.array([80, 20, 0])))

        self.assertEqual(controller._alg_function, boppy.simulators.ssa.SSA)

    def test_application_controller_simulation(self):
        controller = boppy.core.MainController(self.raw_input)

        population, times = controller.simulate()

        self.assertEqual(population.shape[1], len(self.raw_input['Species']))

        self.assertEqual(True, np.allclose(population[-2:],
                                           np.array([3, 12, 85, 3, 11, 86]).reshape(2, 3)))
        self.assertEqual(True, np.allclose(times[-2:], np.array([99.01631504651024,
                                                                 100.12470283342867])))

    def tearDown(self):
        np.random.seed()
