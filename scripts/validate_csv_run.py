import argparse
from diy_automl.validation import validate
p=argparse.ArgumentParser();p.add_argument("directory");a=p.parse_args()
validate(a.directory,2)
print("Valid schema 2 run")
