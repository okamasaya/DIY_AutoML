import argparse
from diy_automl.tuning_validation import validate_tuning

p=argparse.ArgumentParser(description='Validate schema 7 without classifier refitting')
p.add_argument('directory')
p.add_argument('--allow-failed',action='store_true')
a=p.parse_args()
m=validate_tuning(a.directory,allow_failed=a.allow_failed)
print(f"Valid schema 7 SL tuning: {m['status']}")
