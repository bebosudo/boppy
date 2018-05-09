import yaml


def converter_yaml_string_to_dict(yaml_string):
    return yaml.load(yaml_string)
