import yaml


def yaml_string_to_dict_converter(yaml_string):
    return yaml.load(yaml_string)


def filename_to_dict_converter(filename):
    with open(filename) as input_fd:
        return yaml_string_to_dict_converter(input_fd.read())
