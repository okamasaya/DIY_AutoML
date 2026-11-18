"""Validate a CSV audit, optionally recomputing its profile from the source CSV."""

import argparse

from diy_automl.data_audit_validation import validate_data_audit


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("directory")
parser.add_argument("--csv", help="Original CSV to verify and re-scan")
args = parser.parse_args()
validate_data_audit(args.directory, csv=args.csv)
print("Valid data audit")
