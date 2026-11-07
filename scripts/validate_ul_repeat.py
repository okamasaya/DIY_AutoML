import argparse
from diy_automl.validation import validate
p=argparse.ArgumentParser();p.add_argument("directory");a=p.parse_args()
validate(a.directory,4)
print("Valid schema 4 UL repeat")
