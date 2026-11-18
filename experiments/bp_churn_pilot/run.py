"""Run the BP data preparation and model comparison on the UCI Churn CSV."""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import os
from pathlib import Path
from threading import Event, Thread
from time import perf_counter
from zipfile import ZipFile

from experiments.tabular_scale_benchmark.run import (
    THREAD_VARS, fingerprint, host, measure, read_json, write_json,
)


SOURCE_URL = "https://archive.ics.uci.edu/static/public/563/iranian%2Bchurn%2Bdataset.zip"
CSV_SHA256 = "90d5fb6bd1630cd4de4b4d28fcf8b4cb92a8f6ab7484605b0799d47386f7dbe1"
FEATURES = (
    "Call  Failure", "Complains", "Subscription  Length", "Seconds of Use",
    "Frequency of use", "Frequency of SMS", "Distinct Called Numbers",
)
TARGET = "Churn"
SAMPLE_SECONDS = 0.010


def read_original(archive, output):
    with ZipFile(archive) as zipped:
        output.write_bytes(zipped.read("Customer Churn.csv"))
    if fingerprint(output)["sha256"] != CSV_SHA256:
        raise ValueError("The supplied CSV differs from the benchmark input")


def temporary_bytes(scratch, output):
    paths = [scratch, *output.glob(".conversion-*"), *output.glob(".extract-*")]
    size = 0
    for path in paths:
        for folder, _, names in path.walk():
            for name in names:
                try:
                    size += (folder / name).stat().st_size
                except FileNotFoundError:
                    continue
    return size


def sample_temporary_files(scratch, output, stop, samples):
    while not stop.is_set():
        samples.append(temporary_bytes(scratch, output))
        stop.wait(SAMPLE_SECONDS)


def measure_stage(trial, stage, command, repeat, env):
    scratch = trial / f"{stage}-temp"
    scratch.mkdir()
    output = trial / f"{stage}-{repeat}" / "output"
    child_env = dict(env, TMP=str(scratch), TEMP=str(scratch), TMPDIR=str(scratch))
    stop = Event()
    samples = []
    sampler = Thread(target=sample_temporary_files, args=(scratch, output, stop, samples))
    sampler.start()
    try:
        output, evidence = measure(trial, stage, command, repeat, child_env)
    finally:
        stop.set()
        sampler.join()
    evidence["observed_temporary_bytes"] = max(samples)
    evidence["temporary_samples"] = len(samples)
    return output, evidence


def unique_profiles(source, output):
    groups = defaultdict(list)
    counts = Counter()
    with source.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        if next(reader) != [*FEATURES, TARGET]:
            raise ValueError("Unexpected projection")
        for row in reader:
            values = tuple(map(int, row))
            features, target = values[:-1], values[-1]
            counts[target] += 1
            groups[features].append(target)

    keep = Counter()
    conflict_groups = conflict_rows = consistent_repeats = 0
    with output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow([*FEATURES, TARGET])
        for features, targets in groups.items():
            if len(set(targets)) > 1:
                conflict_groups += 1
                conflict_rows += len(targets)
                continue
            target = targets[0]
            writer.writerow((*features, target))
            keep[target] += 1
            consistent_repeats += len(targets) - 1
    return {
        "projected_counts": dict(counts), "model_input_counts": dict(keep),
        "conflicting_profiles": conflict_groups, "conflicting_rows_removed": conflict_rows,
        "repeated_rows_removed": consistent_repeats,
    }


def summarize_selection(selection, input_csv):
    import numpy as np
    from openpyxl import load_workbook
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    manifest = read_json(selection / "manifest.json")
    if manifest["classes"] != [0, 1] or manifest["input_sha256"] != fingerprint(input_csv)["sha256"]:
        raise ValueError("Unexpected selection input or classes")
    with input_csv.open(encoding="utf-8", newline="") as stream:
        targets = np.array([int(row[TARGET]) for row in csv.DictReader(stream)])
    with np.load(selection / "arrays.npz", allow_pickle=False) as arrays:
        train, truth, prediction = (arrays[key] for key in ("train", "truth", "predictions"))
        majority = np.bincount(targets[train], minlength=2).argmax()

    def scores(pred):
        return {
            "confusion": confusion_matrix(truth, pred, labels=[0, 1]).tolist(),
            "accuracy": accuracy_score(truth, pred),
            "macro_f1": f1_score(truth, pred, labels=[0, 1], average="macro"),
        }

    book = load_workbook(selection / "report.xlsx", read_only=True, data_only=True)
    try:
        intervals = {name: (lower, upper) for name, _, lower, upper in
                     book["SL_Test_Intervals"].iter_rows(min_row=2, values_only=True)}
    finally:
        book.close()
    return {
        "model": manifest["selected_model"], **scores(prediction), "macro_f1_interval": intervals["F1"],
        "baseline_label": int(majority), "baseline": scores(np.full_like(truth, majority)),
    }


def run(archive, output, repeats):
    output.mkdir(parents=True)
    original = output / "original.csv"
    read_original(archive, original)
    env = dict(os.environ, **dict.fromkeys(THREAD_VARS, "1"))
    results = {
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": host(), "source_url": SOURCE_URL, "archive": fingerprint(archive),
        "original": fingerprint(original), "features": list(FEATURES),
        "repeats": repeats, "temporary_sample_seconds": SAMPLE_SECONDS, "trials": [],
    }
    source_args = ["prepare-bp", "--csv", str(original), "--target", TARGET]
    for repeat in range(1, repeats + 1):
        trial = output / f"trial-{repeat}"
        trial.mkdir()
        metrics = {}
        _, metrics["audit"] = measure_stage(trial, "audit", source_args, repeat, env)
        compressed, metrics["audit_compress"] = measure_stage(
            trial, "compress", [*source_args, "--compress"], repeat, env,
        )
        parquet = compressed / "converted" / "data.parquet"
        projection, metrics["extract"] = measure_stage(
            trial, "extract", ["extract-data", "--source", str(parquet), "--columns", *FEATURES, TARGET], repeat, env,
        )
        model_input = trial / "model-input.csv"
        start = perf_counter()
        cohorts = unique_profiles(projection / "data.csv", model_input)
        cohort_seconds = perf_counter() - start
        _, metrics["recheck"] = measure_stage(
            trial, "recheck", ["prepare-bp", "--csv", str(model_input), "--target", TARGET], repeat, env,
        )
        selection, metrics["select"] = measure_stage(
            trial, "select", ["select", "--csv", str(model_input), "--target", TARGET,
                              "--seed", "42", "--bootstrap-repeats", "1000"], repeat, env,
        )
        results["trials"].append({
            "repeat": repeat, "measurements": metrics, "cohort_preparation_seconds": cohort_seconds,
            "cohorts": cohorts, "parquet": fingerprint(parquet), "model_input": fingerprint(model_input),
            "selection": summarize_selection(selection, model_input),
        })
        write_json(output / "results.json", results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if os.name != "nt" or args.repeats < 1:
        parser.error("Use Windows and a positive repeat count")
    run(args.archive.resolve(), args.output.resolve(), args.repeats)


if __name__ == "__main__":
    main()
