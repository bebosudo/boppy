#!/usr/bin/env python3

import unittest

import io
from .context import poppy


class YAMLTester(unittest.TestCase):
    """Test that the YAML converter meets the YAML standard."""

    def setUp(self):
        # Simulate the behaviour of a file from a string.
        self.fake_input_file = io.StringIO("Species:\n  - x\n  - y\n  - z\n\nObservables:\n"
                                           "  - x_tot = x + y\n\nReactions:\n  - 3x + y = z")
        self.expected_dictionary = {'Species': ['x', 'y', 'z'],
                                    'Observables': ['x_tot = x + y'],
                                    'Reactions': ['3x + y = z']}

    def test_interpret_file_from_yaml_to_dict(self):
        self.assertTrue(self.expected_dictionary ==
                        poppy.file_parser.converter_yaml_string_to_dict(self.fake_input_file))

    def test_interpret_empty_file_from_yaml_to_dict(self):
        empty_input_file = io.StringIO("")
        self.assertIsNone(poppy.file_parser.converter_yaml_string_to_dict(empty_input_file))

    def tearDown(self):
        # Here we can place repetitive code that should be performed _after_ every test is executed.
        pass


if __name__ == '__main__':
    unittest.main()
