import argparse
import os.path
from file_parser import filename_to_dict_converter

def does_file_exist(parser, arg):
	if not os.path.exists(arg):
		parser.error("The file %s does not exist!" % arg)
	else:
		filename_to_dict_converter(arg)

parser = argparse.ArgumentParser(description="...")
parser.add_argument("file", help="input file")
args = parser.parse_args()
out = args.file
does_file_exist(parser, out)