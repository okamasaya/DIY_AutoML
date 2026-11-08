import argparse
from diy_automl.ul_subsample_validation import validate_ul_subsample

p=argparse.ArgumentParser(description='Validate schema 6 without fitting a clustering model')
p.add_argument('directory')
p.add_argument('--allow-failed',action='store_true')
a=p.parse_args()
m=validate_ul_subsample(a.directory,allow_failed=a.allow_failed)
print(f"Valid schema 6 UL subsampling: {m['status']}")
