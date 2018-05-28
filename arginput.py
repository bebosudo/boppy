import argparse 

parser = argparse.ArgumentParser(description="...")
parser.add_argument("file", help="input file")
args = parser.parse_args()
out = args.file
print(out)