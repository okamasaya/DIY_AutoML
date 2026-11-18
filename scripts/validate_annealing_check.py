"""Verify a saved annealing check without retraining."""

import argparse

from diy_automl.annealing_validation import validate_annealing_check


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("directory")
args = parser.parse_args()
validate_annealing_check(args.directory)
print("Valid annealing check")
