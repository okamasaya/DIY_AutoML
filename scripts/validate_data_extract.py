"""Check an extracted dataset, optionally verifying its source file."""

import argparse

from diy_automl.data_extract_validation import validate_data_extract


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("directory")
parser.add_argument("--source", help="Original CSV, Parquet or DuckDB file")
args = parser.parse_args()
validate_data_extract(args.directory, source=args.source)
print("Valid data extraction")
