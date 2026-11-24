"""Replay a saved comparison of the extended and unscented Kalman filters."""

import argparse

from diy_automl.ekf_benchmark import COMPARISON_SCHEMA
from diy_automl.validation import validate


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("directory")
args = parser.parse_args()
validate(args.directory, COMPARISON_SCHEMA)
print("Valid EKF and UKF benchmark")
