"""Check a saved uncertainty report without model training or TensorFlow."""
import argparse
from diy_automl.uncertainty_validation import validate_uncertainty

p=argparse.ArgumentParser()
p.add_argument('directory')
a=p.parse_args()
print(validate_uncertainty(a.directory)['status'])
