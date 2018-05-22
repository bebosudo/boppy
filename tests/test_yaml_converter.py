#!/usr/bin/env python3

from . import context
import unittest
import poppy.file_parser

import os.path
from tempfile import TemporaryDirectory


class YAMLTester(unittest.TestCase):
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
