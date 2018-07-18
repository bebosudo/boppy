#!/usr/bin/env python3

import argparse
import os.path
import yaml
import logging

from boppy.application import MainController
from boppy.utils.input_loading import filename_to_dict_converter

LOGGER = logging.getLogger(__name__)


def _is_valid_yaml(parser, filename):
    try:
        if not os.path.exists(filename):
            parser.error("cannot access '{}': No such file or directory".format(filename))
        return filename_to_dict_converter(filename)
    except yaml.scanner.ScannerError as exc:
        LOGGER.error("Unable to correctly interpret the input file '%s'; "
                     "traceback:\n%s\nexiting...", filename, str(exc))


def main():
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument("-a", "--alg_file", help="algorithm details input yaml", required=True,
                        type=lambda f: _is_valid_yaml(parser, f))
    parser.add_argument("-s", "--simul_file", help="simulation details input yaml", required=True,
                        type=lambda f: _is_valid_yaml(parser, f))
    # here we want also an optional `--verbosity' parameter
    args = parser.parse_args()

    return MainController(args.alg_file, args.simul_file)


if __name__ == "__main__":
    controller = main()
    controller.simulate()

    from pprint import pprint
    pprint(controller.simulation_out_population)
    pprint(controller.simulation_out_times)
