from . import context
import unittest
import poppy.core
import poppy.file_parser

import os.path
from tempfile import TemporaryDirectory
import numpy as np


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
                        poppy.file_parser.yaml_string_to_dict_converter(self.example_input_text))

    def test_interpret_empty_file_from_yaml_to_dict(self):
        empty_input_file = ""
        self.assertIsNone(poppy.file_parser.yaml_string_to_dict_converter(empty_input_file))

    def test_filename_to_dict(self):
        with TemporaryDirectory() as tmpdirname:
            temp_filename = os.path.join(tmpdirname, "test_input.txt")
            with open(temp_filename, "w") as temp_fd:
                temp_fd.write(self.example_input_text)

            self.assertTrue(self.expected_converted_dictionary ==
                            poppy.file_parser.filename_to_dict_converter(temp_filename))

    def test_empty_filename(self):
        with TemporaryDirectory() as tmpdirname:
            temp_filename = os.path.join(tmpdirname, "test_input.txt")
            with open(temp_filename, "w") as temp_fd:
                temp_fd.write("")
            self.assertIsNone(poppy.file_parser.filename_to_dict_converter(temp_filename))

            with open(temp_filename, "w") as temp_fd:
                temp_fd.write("\n")
            self.assertIsNone(poppy.file_parser.filename_to_dict_converter(temp_filename))

    def test_non_existing_filename(self):
        with TemporaryDirectory() as tmpdirname:
            temp_filename = os.path.join(tmpdirname, "test_input.txt")
            with self.assertRaises(FileNotFoundError):
                poppy.file_parser.filename_to_dict_converter(temp_filename)

    def tearDown(self):
        # Here we can place repetitive code that should be performed _after_ every test is executed.
        pass


class ParserTest(unittest.TestCase):
    """Test that the parser is able to correctly convert reactions and rate functions."""

    def setUp(self):
        self.input_data = {"Species": poppy.core.VariableCollection(["x_s", "x_i", "x_r"]),
                           "Parameters": poppy.core.ParameterCollection({'k_i': 1,
                                                                         'k_r': 0.05,
                                                                         'k_s': 0.01}),
                           "Rate functions": ["3 + 4 * 2 / ( 1 - 5 )",
                                              # "k_i * x_i * x_s / N",
                                              "-3* 2 * k_i * x_i * x_s / sum(x_s + 2 * x_i - x_i + x_r)",
                                              "k_i * x_i * x_s /    N  "],
                           "Reactions": ["x_s => x_i",
                                         "x_s + x_i => x_r",
                                         "3x_s + x_i => x_r"],
                           "Initial conditions": [80, 20, 0],
                           }
        self.expected_update_vector = [np.array([-1, 1, 0]),
                                       np.array([-1, -1, 1]),
                                       np.array([-3, -1, 1])]

    def test_convert_simplest_reaction(self):
        reaction_obj = poppy.core.Reaction(self.input_data["Reactions"][0],
                                           self.input_data["Species"])

        self.assertEqual(np.array_equiv(reaction_obj.update_vector,
                                        self.expected_update_vector[0]), True)

    def test_convert_basic_reaction(self):
        reaction_obj = poppy.core.Reaction(self.input_data["Reactions"][1],
                                           self.input_data["Species"])

        self.assertEqual(np.array_equiv(reaction_obj.update_vector,
                                        self.expected_update_vector[1]), True)

    def test_convert_normal_reaction(self):
        reaction_obj = poppy.core.Reaction(self.input_data["Reactions"][2],
                                           self.input_data["Species"])

        self.assertEqual(np.array_equiv(reaction_obj.update_vector,
                                        self.expected_update_vector[2]), True)

    def test_missing_value_raises_exc(self):
        with self.assertRaisesRegex(ValueError, "Unable to find reagent '.*' inside the list of variables"):
            poppy.core.Reaction("R1 + R2 => 2 P1", self.input_data["Species"])

    def test_reaction_collection(self):
        reaction_collection = poppy.core.ReactionCollection(self.input_data["Reactions"],
                                                            self.input_data["Species"])

        for reac, upd_vec in zip(reaction_collection, self.expected_update_vector):
            self.assertEqual(np.array_equiv(reac.update_vector, upd_vec), True)

    def test_basic_rate_function(self):
        rate_func = poppy.core.RateFunction(self.input_data["Rate functions"][0],
                                            self.input_data["Species"],
                                            self.input_data["Parameters"])

    def test_normal_rate_function(self):
        rate_func = poppy.core.RateFunction(self.input_data["Rate functions"][1],
                                            self.input_data["Species"],
                                            self.input_data["Parameters"])

        # rate_func(self.input_data["Initial conditions"])
