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
        self.example_input_text = ("Species:\n  - x\n  - y\n  - z\n\nObservables:\n"
                                   "  - x_tot = x + y\n\nReactions:\n  - 3x + y = z\n")
        self.expected_converted_dictionary = {"Species": ["x", "y", "z"],
                                              "Observables": ["x_tot = x + y"],
                                              "Reactions": ["3x + y = z"]}

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
        self.input_data = {"Species": [poppy.core.Variable("x"),
                                       poppy.core.Variable("y"),
                                       poppy.core.Variable("z")],
                           "Reactions": ["x + y => z",
                                         "3x + y => z"]}
        self.expected_update_vector = [np.array([-1, -1, 1]), np.array([-3, -1, 1])]

    def test_convert_basic_reaction(self):
        reaction_obj = poppy.core.Reaction(self.input_data["Reactions"][0],
                                           self.input_data["Species"])

        self.assertEqual(np.array_equiv(reaction_obj.update_vector,
                                        self.expected_update_vector[0]), True)

    def test_convert_normal_reaction(self):
        reaction_obj = poppy.core.Reaction(self.input_data["Reactions"][1],
                                           self.input_data["Species"])

        self.assertEqual(np.array_equiv(reaction_obj.update_vector,
                                        self.expected_update_vector[1]), True)

    def test_missing_value_raises_exc(self):
        with self.assertRaisesRegex(ValueError, "Unable to find reagent '.*' inside the list of variables"):
            poppy.core.Reaction("R1 + R2 => 2 P1", self.input_data["Species"])
