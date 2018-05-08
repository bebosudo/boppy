import yaml

with open("../example/test_input.yaml") as input_file_fd:
    print(yaml.load("".join(input_file_fd.readlines())))
    #
