#!/usr/bin/env python3


import unittest
import io
import os.path
from tempfile import TemporaryDirectory

from .context import poppy


class YAMLTester(unittest.TestCase):
    """Test that the YAML converter meets the YAML standard."""

    def setUp(self):
        self.example_input_file = ("Species:\n  - x\n  - y\n  - z\n\nObservables:\n"
                                   "  - x_tot = x + y\n\nReactions:\n  - 3x + y = z\n")
        # Simulate the behaviour of a file from a string.
        self.fake_input_file = io.StringIO(self.example_input_file)
        self.expected_dictionary = {"Species": ["x", "y", "z"],
                                    "Observables": ["x_tot = x + y"],
                                    "Reactions": ["3x + y = z"]}

    def test_interpret_file_from_yaml_to_dict(self):
        self.assertTrue(self.expected_dictionary ==
                        poppy.file_parser.yaml_string_to_dict_converter(self.fake_input_file))

    def test_interpret_empty_file_from_yaml_to_dict(self):
        empty_input_file = io.StringIO("")
        self.assertIsNone(poppy.file_parser.yaml_string_to_dict_converter(empty_input_file))

    def test_filename_to_dict(self):
        with TemporaryDirectory() as tmpdirname:
            temp_filename = os.path.join(tmpdirname, "test_input.txt")
            with open(temp_filename, "w") as temp_fd:
                temp_fd.write(self.example_input_file)

            self.assertTrue(self.expected_dictionary ==
                            poppy.file_parser.filename_to_dict_converter(temp_filename))

    def tearDown(self):
        # Here we can place repetitive code that should be performed _after_ every test is executed.
        pass


if __name__ == "__main__":
    unittest.main()
