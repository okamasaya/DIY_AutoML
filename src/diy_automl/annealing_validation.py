"""Replay saved feature selection and reports without fitting any classifiers."""

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .annealing import EXHAUSTIVE, columns, search_subsets
from .annealing_workflow import DIAGNOSTICS, SCHEMA, annealing_settings, report_tables, search_plan, subset_score
from .sl_ranking import cv_assignments, evaluation_split
from .sl_uncertainty_audit import parse_offsets
from .sl_uncertainty_audit_validation import validate_metric_roundoff
from .ul_subsample_validation import require, same_table


def verify_predictions(values, shape, classes, name):
    require(values.shape == shape and np.issubdtype(values.dtype, np.integer) and np.isin(values, classes).all(),
            f"Invalid shape or class labels: {name}")


def replay_search(arrays, settings, space, methods, truth, assignments, classes):
    count = 1 + 2 * min(space.size, settings["budget"]) + (space.size if EXHAUSTIVE in methods else 0)
    masks, predictions, times = (arrays[key] for key in ("cv_masks", "cv_predictions", "cv_seconds"))
    require(masks.shape == (count, space.width) and masks.dtype == bool, "Invalid CV subset masks")
    verify_predictions(predictions, (count, *assignments.shape), classes, "cv_predictions")
    require(times.shape == (count,) and np.isfinite(times).all() and (times >= 0).all(), "Invalid CV times")

    evidence = iter(zip(masks, predictions, times))

    def evaluate(mask):
        stored_mask, predicted, elapsed = next(evidence)
        require(np.array_equal(np.flatnonzero(stored_mask), columns(mask, space.width)), "CV proposal differs")
        return subset_score(truth, classes, assignments, predicted, settings), float(elapsed)

    return {method: search_subsets(space, method, settings, evaluate) for method in methods}


def verify_holdout(arrays, manifest, histories, classes):
    features, settings = manifest["features"], manifest["settings"]
    n_test = len(arrays["test"])
    predictions, times = arrays["test_predictions"], arrays["test_seconds"]
    verify_predictions(predictions, (len(histories), n_test), classes, "test_predictions")
    require(times.shape == (len(histories),) and np.isfinite(times).all() and (times >= 0).all(), "Invalid refit times")
    checked = {"test_predictions", "test_seconds"}
    for index, history in enumerate(histories.values()):
        names = [features[i] for i in columns(history[history[-1].best].mask, len(features))]
        rounded, roundoff, offsets = [f"test_{index}_{key}" for key in DIAGNOSTICS]
        require((rounded in arrays) == (roundoff in arrays), "Incomplete Float32 evidence")
        if rounded in arrays:
            verify_predictions(arrays[rounded], (n_test,), classes, rounded)
            values = arrays[roundoff]
            require(values.shape == (len(names),) and np.isfinite(values).all() and (values >= 0).all(),
                    "Invalid Float32 roundoff")
            checked.update((rounded, roundoff))
        scenarios = settings["shifts"]
        if scenarios:
            verify_predictions(arrays[offsets], (len(scenarios), n_test), classes, offsets)
            for row, scenario in enumerate(scenarios):
                if scenario["feature"] not in names:
                    require(np.array_equal(arrays[offsets][row], predictions[index]),
                            "Offset on an excluded feature changed predictions")
            checked.add(offsets)
    return checked


def validate_annealing_check(directory):
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text())
    require(manifest["schema_version"] == SCHEMA and manifest["status"] == manifest["stage"] == "completed",
            "Annealing check did not complete")
    require(set(manifest["artifacts"]) == {"report.xlsx", "arrays.npz"}, "Unexpected artifact set")
    for name, checksum in manifest["artifacts"].items():
        require(sha256((root / name).read_bytes()).hexdigest() == checksum, f"Artifact checksum mismatch: {name}")

    settings, features = manifest["settings"], manifest["features"]
    args = {key: value for key, value in settings.items() if key not in ("folds", "comparison_fraction", "tie_atol")}
    args["shifts"] = parse_offsets(
        [f"{scenario['feature']}={scenario['offset']!r}" for scenario in settings["shifts"]], features,
    )
    require(settings == annealing_settings(width=len(features), **args), "Annealing settings differ")
    space, methods = search_plan(len(features), settings)
    require(manifest["methods"] == methods, "Search arms differ")
    with np.load(root / "arrays.npz", allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
    target = arrays["target"]
    classes = np.arange(len(manifest["classes"]))
    require(target.ndim == 1 and np.issubdtype(target.dtype, np.integer)
            and np.array_equal(np.unique(target), classes), "Invalid encoded target")
    train, _, test = evaluation_split(target, classes, settings)
    assignments = cv_assignments(target[train], settings)
    for key, expected in (("train", train), ("test", test), ("cv_fold_assignments", assignments)):
        require(np.issubdtype(arrays[key].dtype, np.integer) and np.array_equal(arrays[key], expected), f"{key} differs")

    histories = replay_search(arrays, settings, space, methods, target[train], assignments, classes)
    checked = verify_holdout(arrays, manifest, histories, classes)
    expected_tables, evidence = report_tables(histories, arrays, manifest, space)
    for name, expected in evidence.items():
        require(np.array_equal(arrays[name], expected), f"Bootstrap evidence differs: {name}")
    checked.update(evidence)
    checked.update(("target", "train", "test", "cv_fold_assignments", "cv_masks", "cv_predictions", "cv_seconds"))
    require(set(arrays) == checked, "Annealing array set differs")
    columns_as_text = ("Proposal", "Feature", "Actual Class", "Class", "Feature With Largest Roundoff", "Exact Fraction")
    actual_tables = pd.read_excel(root / "report.xlsx", sheet_name=None, keep_default_na=False,
                                  dtype=dict.fromkeys(columns_as_text, str))
    require(set(actual_tables) == set(expected_tables), "Annealing sheet names differ")
    require(manifest["sheets"] == {name: len(frame) for name, frame in expected_tables.items()},
            "Annealing sheet row counts differ")
    for name, expected in expected_tables.items():
        same_table(actual_tables[name], expected.fillna(""), f"Annealing table differs: {name}")
    validate_metric_roundoff(actual_tables["Numerical_Metric"], expected_tables["Numerical_Metric"])
    return manifest
