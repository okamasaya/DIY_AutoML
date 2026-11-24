"""Replay a saved particle filter comparison."""

import argparse

from diy_automl.ekf_benchmark import PARTICLE_SCHEMA
from diy_automl.validation import validate


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("directory")
args = parser.parse_args()
validate(args.directory, PARTICLE_SCHEMA)
print("Valid particle filter benchmark")
