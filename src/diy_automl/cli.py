import argparse
from datetime import datetime,timezone
import json
from .workflow import run_csv
from .selection_workflow import select_csv
from .clustering_repeat_workflow import repeat_clustering_csv
from .deep_learning_repeat_workflow import repeat_deep_learning_csv
from .clustering_subsample_workflow import subsample_clustering_csv
from .tuning_workflow import tune_csv
from .excel_report import render_excel_report
from .uncertainty_workflow import evaluate_uncertainty
from .sl_ranking import rank_sl_csv
from .sl_uncertainty_audit import audit_sl_uncertainty
from .annealing_workflow import annealing_check_csv
from .data_audit import audit_data_csv, prepare_bp_csv
from .data_conversion import FORMATS, TYPE_MODES
from .data_extract import extract_data
from .supervised import METRICS, candidates

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
            if command=="select":
                grouping=p.add_mutually_exclusive_group()
                grouping.add_argument("--group-column",help="Keep all rows with the same ID together and exclude this column from features")
                grouping.add_argument("--group-identical",action="store_true",help="Keep exactly matching numeric inputs together")
                grouping.add_argument("--time-column",help="Use ISO 8601 observation dates for chronological selection and holdout")
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
    p=sub.add_parser("rank-sl",help="Rank the five fixed classifiers on shared repeated cross-validation")
    p.add_argument("--csv",required=True)
    p.add_argument("--target",required=True)
    p.add_argument("--output",default=None)
    p.add_argument("--seed",type=int,default=42,help="Seed for data partitions")
    p.add_argument("--model-seed",type=int,default=42,help="Fixed model seed across CV repeats")
    p.add_argument("--cv-repeats",type=int,default=5)
    p.add_argument("--metric",choices=METRICS,default="F1")
    p.add_argument("--comparison-fraction",type=float,default=0.0,
                   help="Share of the non-test pool reserved for a separate paired comparison")
    p.add_argument("--bootstrap-repeats",type=int,default=1000)
    p.add_argument("--bootstrap-seed",type=int,default=43)
    p.add_argument("--confidence",type=float,default=.95)
    p.add_argument("--shift",action="append",default=[],metavar="FEATURE=OFFSET",
                   help="Hypothetical constant test input offset for the selected model, repeatable")
    p=sub.add_parser("annealing-check",help="Compare CPU simulated annealing and random feature subset search")
    p.add_argument("--csv",required=True)
    p.add_argument("--target",required=True)
    p.add_argument("--output",default=None)
    p.add_argument("--seed",type=int,default=42,help="Seed for data partitions")
    p.add_argument("--model",choices=[name for name, _ in candidates()],default="SVM")
    p.add_argument("--model-seed",type=int,default=42)
    p.add_argument("--cv-repeats",type=int,default=2)
    p.add_argument("--metric",choices=METRICS,default="F1")
    p.add_argument("--budget",type=int,default=32,help="Maximum distinct subsets per search arm")
    p.add_argument("--max-features",type=int,help="Maximum columns in searched subsets")
    p.add_argument("--search-seed",type=int,default=44)
    p.add_argument("--initial-temperature",type=float,default=.03)
    p.add_argument("--final-temperature",type=float,default=.002)
    p.add_argument("--exhaustive",action="store_true",help="Add an exhaustive check for a small subset space")
    p.add_argument("--bootstrap-repeats",type=int,default=1000)
    p.add_argument("--bootstrap-seed",type=int,default=43)
    p.add_argument("--confidence",type=float,default=.95)
    p.add_argument("--shift",action="append",default=[],metavar="FEATURE=OFFSET",
                   help="Hypothetical constant test input offset, repeatable")
    p=sub.add_parser("audit-data",help="Inspect a CSV and record evaluation context before training")
    p.add_argument("--csv",required=True)
    p.add_argument("--output",default=None)
    p.add_argument("--target",help="Class label column")
    p.add_argument("--id-column",help="Identifier used to count groups")
    p.add_argument("--time-column",help="Timestamp column containing ISO 8601 values")
    p.add_argument("--prediction-goal",help="What will be predicted")
    p.add_argument("--prediction-time",help="When a prediction would be made")
    p.add_argument("--row-unit",help="What one input row represents")
    p.add_argument("--label-source",help="How the class labels were obtained")
    p.add_argument("--convert",action="append",choices=FORMATS,default=[],
                   help="Export to a binary format after auditing, repeatable")
    p.add_argument("--conversion-types",choices=TYPE_MODES,default="preserve",
                   help="Preserve CSV text or convert eligible numeric feature columns exactly")
    p=sub.add_parser("prepare-bp",help="点検結果を日本語で確認し、必要に応じて作業用ファイルを圧縮します")
    p.add_argument("--csv",required=True,help="元のCSVファイル")
    p.add_argument("--target",help="正解ラベルの列")
    p.add_argument("--compress",action="store_true",help="作業用の圧縮Parquetを追加で保存")
    p.add_argument("--output",default=None,help="結果を保存する新しいフォルダ")
    p=sub.add_parser("bp-preview",help="CSVを選び、学習前のデータ点検を画面で確認します")
    p.add_argument("--csv",help="起動時に選ぶCSVファイル")
    p.add_argument("--output-dir",default="runs/bp-preview",help="点検結果を保存する親フォルダ")
    p=sub.add_parser("extract-data",help="Select columns from a CSV, Parquet or DuckDB file")
    p.add_argument("--source",required=True,help="Path to a .csv, .parquet or .duckdb file")
    p.add_argument("--columns",nargs="+",required=True,metavar="COLUMN",help="Output columns in the requested order")
    p.add_argument("--where",action="append",nargs=3,default=[],metavar=("COLUMN", "OP", "VALUE"),
                   help="Filter Parquet or DuckDB using eq, ne, lt, le, gt or ge. Repeat to apply all conditions.")
    p.add_argument("--limit",type=int,help="Maximum output rows in source order")
    p.add_argument("--output",default=None)
    p=sub.add_parser("report")
    p.add_argument("--run",dest="source",required=True)
    p.add_argument("--output",default=None)
    p.add_argument("--csv",default=None)
    p.add_argument("--max-models",type=int,default=3)
    p=sub.add_parser("uncertainty",help="Evaluate fixed-model metric sampling uncertainty from a saved classification run")
    p.add_argument("--run",dest="source",required=True)
    p.add_argument("--output",default=None)
    p.add_argument("--csv",default=None,help="Original hash-matching CSV if a historical run lacks target labels")
    p.add_argument("--bootstrap-repeats",type=int,default=1000)
    p.add_argument("--bootstrap-seed",type=int,default=43)
    p.add_argument("--confidence",type=float,default=.95)
    p.add_argument("--resampling",choices=["stratified","ordinary"],default="stratified")
    p=sub.add_parser("audit-uncertainty",help="Audit statistical uncertainty, numerical effects and test input offsets")
    p.add_argument("--run",dest="source",required=True,help="Completed rank-sl output")
    p.add_argument("--csv",required=True,help="Original hash-matching CSV")
    p.add_argument("--output",default=None)
    p.add_argument("--shift",action="append",default=[],metavar="FEATURE=OFFSET",
                   help="Hypothetical constant test input offset, repeatable")
    args=vars(parser.parse_args()); command=args.pop("command")
    if command=="bp-preview":
        from .bp_preview import launch_preview

        try:
            launch_preview(**args)
        except RuntimeError as error:
            parser.exit(2, f"{error}\n")
        return
    if args["output"] is None: args["output"]="runs/"+command+"-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    handlers={
        "run":run_csv,"select":select_csv,"repeat-ul":repeat_clustering_csv,
        "repeat-dl":repeat_deep_learning_csv,"subsample-ul":subsample_clustering_csv,
        "tune":tune_csv,"rank-sl":rank_sl_csv,"report":render_excel_report,
        "uncertainty":evaluate_uncertainty,"audit-uncertainty":audit_sl_uncertainty,
        "annealing-check":annealing_check_csv,"audit-data":audit_data_csv,"extract-data":extract_data,
        "prepare-bp":prepare_bp_csv,
    }
    result = handlers[command](**args)
    print(result, flush=True)
    if (command == "audit-data" and args["convert"]) or (command == "prepare-bp" and args["compress"]):
        conversion = json.loads((result / "manifest.json").read_text(encoding="utf-8"))["conversion"]
        if conversion["status"] != "completed":
            parser.exit(2, f"Conversion {conversion['status']}: {conversion['reason']}\n")
