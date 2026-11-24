"""Replay a saved extended Kalman filter benchmark."""

import argparse

from diy_automl.ekf_benchmark import SCHEMA
from diy_automl.validation import validate


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("directory")
args = parser.parse_args()
validate(args.directory, SCHEMA)
print("Valid EKF benchmark")
