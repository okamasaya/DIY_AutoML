"""Check an extracted CSV, optionally verifying its intermediate input."""

import argparse

from diy_automl.data_extract_validation import validate_data_extract


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("directory")
parser.add_argument("--source", help="Original Parquet or DuckDB file")
args = parser.parse_args()
validate_data_extract(args.directory, source=args.source)
print("Valid data extraction")
