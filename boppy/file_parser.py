import logging

import yaml

LOGGER = logging.getLogger(__name__)


def yaml_string_to_dict_converter(yaml_string):
    LOGGER.debug("Converting YAML string '%s' to a python dict.",
                 yaml_string[:50].replace("\n", "\\n"))
    return yaml.load(yaml_string)


def filename_to_dict_converter(filename):
    LOGGER.debug("Going to open file '%s' in order to convert it to a python dict.", filename)
    try:
        with open(filename) as input_fd:
            LOGGER.debug("File '%s' opened, going to convert it to a python dict.", filename)
            return yaml_string_to_dict_converter(input_fd.read())

    except FileNotFoundError as e:
        LOGGER.error("Unable to find file '%s' to be converted to a python dict.", filename)
        raise
