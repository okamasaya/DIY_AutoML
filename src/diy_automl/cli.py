import argparse
from datetime import datetime,timezone
from .workflow import run_csv
from .selection_workflow import select_csv
from .clustering_repeat_workflow import repeat_clustering_csv
from .deep_learning_repeat_workflow import repeat_deep_learning_csv
from .clustering_subsample_workflow import subsample_clustering_csv
from .tuning_workflow import tune_csv

def main():
    parser=argparse.ArgumentParser(description="Reimplemented DIY AutoML evaluation")
    sub=parser.add_subparsers(dest="command",required=True)
    for command in ("run","select","repeat-ul","repeat-dl","subsample-ul","tune"):
        p=sub.add_parser(command)
        p.add_argument("--csv",required=True); p.add_argument("--target",required=True)
        p.add_argument("--output",default=None)
        if command in ("run","select","tune"): p.add_argument("--seed",type=int,default=42)
        if command=="run":
            p.add_argument("--methods",nargs="+",choices=["sl","ul","dl"],default=["sl","ul","dl"])
            p.add_argument("--epochs",type=int,default=100)
            p.add_argument("--split-seed",type=int); p.add_argument("--training-seed",type=int)
        elif command in ("select","tune"):
            p.add_argument("--metric",choices=["Accuracy","Precision","Recall","F1"],default="F1")
            p.add_argument("--bootstrap-repeats",type=int,default=1000)
            if command=="tune":
                p.add_argument("--bootstrap-seed",type=int,default=43)
                p.add_argument("--confidence",type=float,default=.95)
        elif command=="repeat-dl":
            p.add_argument("--training-seeds",nargs="+",type=int,default=list(range(42,52)))
            p.add_argument("--split-seed",type=int,default=42)
            p.add_argument("--epochs",type=int,default=100)
            p.add_argument("--batch-size",type=int,default=16)
        elif command=="subsample-ul":
            p.add_argument("--sample-seeds",nargs="+",type=int,default=list(range(42,52)))
            p.add_argument("--fraction",type=float,default=.8)
            p.add_argument("--model-seed",type=int,default=42)
            p.add_argument("--clusters",nargs="+",type=int,default=list(range(2,11)))
            p.add_argument("--min-overlap",type=int,default=10)
        else: p.add_argument("--seeds",nargs="+",type=int,default=list(range(42,52)))
    args=vars(parser.parse_args()); command=args.pop("command")
    if args["output"] is None: args["output"]="runs/"+command+"-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    print({"run":run_csv,"select":select_csv,"repeat-ul":repeat_clustering_csv,"repeat-dl":repeat_deep_learning_csv,"subsample-ul":subsample_clustering_csv,"tune":tune_csv}[command](**args))
