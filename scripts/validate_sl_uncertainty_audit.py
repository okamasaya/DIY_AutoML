"""Verify a saved uncertainty audit without refitting the model."""

import argparse

from diy_automl.sl_uncertainty_audit_validation import validate_sl_uncertainty_audit


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("directory")
args = parser.parse_args()
validate_sl_uncertainty_audit(args.directory)
print("Valid SL uncertainty audit")
