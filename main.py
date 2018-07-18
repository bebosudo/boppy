#!/usr/bin/env python3

import argparse
import os.path
import yaml
import logging

import boppy.core as core
from boppy.utils.input_loading import filename_to_dict_converter

LOGGER = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument("file", help="input file")
    # here we want also an optional `--verbosity' parameter
    args = parser.parse_args()

    if not os.path.exists(args.file):
        parser.error("The file '%s' does not exist!" % args.file)

    try:
        yaml_converted_dict = filename_to_dict_converter(args.file)
        return core.MainController(yaml_converted_dict)

    except yaml.scanner.ScannerError as exc:
        LOGGER.error("Unable to correctly interpret the input file '%s'; "
                     "traceback:\n%s\nexiting...", args.file, str(exc))

    raise SystemExit()


if __name__ == "__main__":
    controller = main()
    controller.simulate()

    from pprint import pprint
    pprint(controller.simulation_out_population)
    pprint(controller.simulation_out_times)
