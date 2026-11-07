import argparse
from diy_automl.dl_repeat_validation import validate_dl_repeat

parser = argparse.ArgumentParser(description='Validate schema 5 without retraining')
parser.add_argument('directory')
parser.add_argument('--allow-failed', action='store_true', help='Validate partial failure artifacts')
args = parser.parse_args()
manifest = validate_dl_repeat(args.directory, allow_failed=args.allow_failed)
print(f"Valid schema 5 DL repeat: {manifest['status']}")
