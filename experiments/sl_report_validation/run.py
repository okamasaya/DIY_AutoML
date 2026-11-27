"""Compare fixed and tuned classification reports on three reference datasets."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from diy_automl import audit_data_csv, render_excel_report
from diy_automl.data_audit_validation import validate_data_audit
from diy_automl.excel_report_validation import validate_excel_report
from diy_automl.reporting import file_sha256, write_json
from diy_automl.validation import validate


DATASETS = ("wine", "seeds", "breast_cancer")
MODES = ("none", "summary", "detail")
ROOT = Path(__file__).resolve().parents[2]


def examine_run(source, directory, workflow):
    run = directory / workflow
    command = [sys.executable, "-X", "utf8", "-m", "diy_automl", workflow,
               "--csv", str(source), "--target", "target", "--metric", "F1", "--seed", "42",
               "--bootstrap-repeats", "1000", "--cpu-threads", "1", "--forest-jobs", "1",
               "--progress", "both", "--output", str(run)]
    started = perf_counter()
    with ((directory / f"{workflow}.stdout.txt").open("wb") as stdout,
          (directory / f"{workflow}.stderr.txt").open("wb") as stderr):
        subprocess.run(command, stdout=stdout, stderr=stderr, cwd=ROOT, check=True)
    train_seconds = perf_counter() - started
    record = validate(run, 3 if workflow == "select" else 7)
    if record["input_sha256"] != file_sha256(source):
        raise RuntimeError(f"Source checksum differs for {run}")
    tuned = workflow == "tune"
    mean_sheet = "SL_Search_Mean" if tuned else "SL_CV_Mean"
    fold_sheet = "SL_Search_Folds" if tuned else "SL_CV_Folds"
    with pd.ExcelFile(run / "report.xlsx") as workbook:
        fold_table = workbook.parse(fold_sheet)
        means = workbook.parse(mean_sheet)
        test = workbook.parse("SL_Test").iloc[0]
        interval = workbook.parse("SL_Test_Intervals").set_index("Metric").loc["F1"]
        chosen = record.get("selected_candidate")
        if tuned:
            selected = means.loc[means.Candidate == chosen].iloc[0]
            settings = json.loads(workbook.parse("SL_Search_Settings").set_index("Candidate").loc[chosen, "Search Parameters"])
        else:
            selected = means.loc[means.Model == record["selected_model"]].iloc[0]
            settings = None
    events = [json.loads(line) for line in (run / "progress.jsonl").read_text(encoding="utf-8").splitlines()]
    cv = [row for row in events if row["stage"] == "cv"]
    np.testing.assert_allclose([row["score"] for row in cv], fold_table["F1"], atol=1e-12, rtol=0)
    phases = {row["stage"]: row["elapsed_seconds"] for row in events if row["stage"] != "cv"}

    with np.load(run / "arrays.npz", allow_pickle=False) as arrays:
        fold_keys = [f"fold_{fold}_{part}" for fold in range(5) for part in ("train", "validation")]
        split = {key: arrays[key] for key in ("train", "test", "truth", *fold_keys)}
        predictions = arrays["predictions"]
    truth = split["truth"]
    class_ids = np.arange(len(record["classes"]))
    f1 = f1_score(truth, predictions, labels=class_ids, average="macro", zero_division=0)
    correct = int(np.count_nonzero(truth == predictions))
    np.testing.assert_allclose([f1, f1, correct / len(truth)], [test.F1, interval.Estimate, test.Accuracy], atol=1e-12, rtol=0)

    renderings = {}
    for mode in MODES:
        stage = perf_counter()
        output = render_excel_report(run, directory / f"{workflow}-{mode}", charts=mode)
        render_seconds = perf_counter() - stage
        rendered = validate_excel_report(output)
        expected = {"none": 0, "summary": 2, "detail": 5 if tuned else 4}[mode]
        if rendered["figure_count"] != expected:
            raise RuntimeError(f"Unexpected number of figures in {output}")
        renderings[mode] = {"figures": expected, "workbook_bytes": rendered["workbook_bytes"],
                            "render_seconds": round(render_seconds, 3)}

    result = {
        "model": record["selected_model"], "selected_candidate": chosen, "search_parameters": settings,
        "tied_candidates": record["tied_candidates"], "candidates": len(means), "cv_folds": len(cv),
        "cv_mean_macro_f1": float(selected.F1), "holdout_correct": correct, "holdout_rows": len(truth),
        "holdout_macro_f1": f1, "conditional_95_interval": [float(interval.Lower), float(interval.Upper)],
        "classes": record["classes"], "confusion_matrix": confusion_matrix(truth, predictions, labels=class_ids).tolist(),
        "phase_elapsed_seconds": phases, "cv_elapsed_seconds": round(phases["refit"] - phases["cv_start"], 3),
        "cv_fit_seconds": float(fold_table["Fit Time (s)"].sum()) if tuned else None,
        "cli_seconds": round(train_seconds, 3), "progress_sha256": file_sha256(run / "progress.jsonl"),
        "renderings": renderings,
    }
    return result, split, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New results directory")
    root = parser.parse_args().output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    results = {"results": {}}
    for dataset in DATASETS:
        source = ROOT / "data" / "reference" / f"{dataset}.csv"
        directory = root / dataset
        directory.mkdir()
        audit = audit_data_csv(source, directory / "audit", target="target")
        validate_data_audit(audit, source)
        print(f"{dataset}: data audit verified", flush=True)
        selected, fixed_split, manifest = examine_run(source, directory, "select")
        print(f"{dataset}: select and three report modes verified", flush=True)
        tuned, tuned_split, _ = examine_run(source, directory, "tune")
        for part in fixed_split:
            np.testing.assert_array_equal(fixed_split[part], tuned_split[part])
        results["environment"] = {key: manifest[key] for key in ("version", "python", "platform", "dependencies")}
        results["results"][dataset] = {
            "input_sha256": manifest["input_sha256"], "rows": len(fixed_split["train"]) + len(fixed_split["test"]),
            "features": len(manifest["features"]), "training_rows": len(fixed_split["train"]),
            "same_holdout_indices": True, "select": selected, "tune": tuned,
        }
        write_json(root / "results.json", results)
        print(f"{dataset}: tune and three report modes verified", flush=True)
    print(root / "results.json", flush=True)


if __name__ == "__main__":
    main()
