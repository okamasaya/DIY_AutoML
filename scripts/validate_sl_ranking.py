"""Verify a saved repeated cross-validation ranking and its holdout results."""

import argparse

from diy_automl.sl_ranking_validation import validate_sl_ranking


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("directory")
args = parser.parse_args()
validate_sl_ranking(args.directory)
print("Valid SL ranking")
