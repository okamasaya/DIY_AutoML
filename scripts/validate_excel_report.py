import argparse
from diy_automl.excel_report_validation import validate_excel_report

p=argparse.ArgumentParser(description='Validate saved Excel reports without retraining')
p.add_argument('directory')
a=p.parse_args();m=validate_excel_report(a.directory)
if m['figure_count']:
    print(f"Valid illustrated Excel: {m['figure_count']} images, original cells unchanged")
else:
    print("Valid numerical Excel copy")
