"""Validate a saved Kalman tracking run."""

import argparse

from diy_automl.kalman_workflow import SCHEMA
from diy_automl.validation import validate


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("directory")
args = parser.parse_args()
validate(args.directory, SCHEMA)
print("Valid Kalman tracking run")
