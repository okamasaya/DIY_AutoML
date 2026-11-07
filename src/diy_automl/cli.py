import argparse
from datetime import datetime,timezone
from .workflow import run_csv
from .selection_workflow import select_csv
from .clustering_repeat_workflow import repeat_clustering_csv

def main():
    parser=argparse.ArgumentParser(description="Reimplemented DIY AutoML evaluation")
    sub=parser.add_subparsers(dest="command",required=True)
    for command in ("run","select","repeat-ul"):
        p=sub.add_parser(command)
        p.add_argument("--csv",required=True); p.add_argument("--target",required=True)
        p.add_argument("--output",default=None)
        if command!="repeat-ul": p.add_argument("--seed",type=int,default=42)
        if command=="run":
            p.add_argument("--methods",nargs="+",choices=["sl","ul","dl"],default=["sl","ul","dl"])
            p.add_argument("--epochs",type=int,default=100)
            p.add_argument("--split-seed",type=int); p.add_argument("--training-seed",type=int)
        elif command=="select":
            p.add_argument("--metric",choices=["Accuracy","Precision","Recall","F1"],default="F1")
            p.add_argument("--bootstrap-repeats",type=int,default=1000)
        else: p.add_argument("--seeds",nargs="+",type=int,default=list(range(42,52)))
    args=vars(parser.parse_args()); command=args.pop("command")
    if args["output"] is None: args["output"]="runs/"+command+"-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    print({"run":run_csv,"select":select_csv,"repeat-ul":repeat_clustering_csv}[command](**args))
