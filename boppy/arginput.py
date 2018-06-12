import argparse
import os.path
import yaml
import logging

import core

from file_parser import filename_to_dict_converter
LOGGER = logging.getLogger(__name__)


def does_file_exist(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)


def main():
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument("file", help="input file")
    args = parser.parse_args()
    out = args.file
    does_file_exist(parser, out)

    try:
        yaml_converted_dict = filename_to_dict_converter(out)
        return core.MainController(yaml_converted_dict)

    except FileNotFoundError as exc:
        LOGGER.error("Unable to find file '%s' to be converted to a python dict.", out)
    except yaml.scanner.ScannerError as exc:
        LOGGER.error("Unable to correctly interpret the input file '%s'; "
                     "traceback:\n%s\nexiting...", out, str(exc))

    raise SystemExit()


if __name__ == "__main__":
    main()
