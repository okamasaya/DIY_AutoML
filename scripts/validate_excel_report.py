import argparse
from diy_automl.excel_report_validation import validate_excel_report

p=argparse.ArgumentParser(description='Validate illustrated Excel without retraining')
p.add_argument('directory')
a=p.parse_args();m=validate_excel_report(a.directory)
print(f"Valid illustrated Excel: {m['figure_count']} images, original cells unchanged")
