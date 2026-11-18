"""Recalculate a saved SL ranking without fitting a model."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .sl_ranking import (
    LEGACY_SCHEMA, SCHEMA, UNCERTAINTY_SCHEMA, add_holdout_tables, audit_tables, cv_assignments, cv_tables, evaluation_split,
    ranking_settings, ranking_uncertainty_tables,
)
from .sl_uncertainty_audit_validation import AUDIT_TEXT_COLUMNS, validate_diagnostic_arrays, validate_metric_roundoff
from .supervised import candidates
from .ul_subsample_validation import require, same_table


def validate_sl_ranking(directory):
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text())
    schema = manifest["schema_version"]
    require(schema in (LEGACY_SCHEMA, UNCERTAINTY_SCHEMA, SCHEMA) and manifest["status"] == "completed"
            and manifest["stage"] == "completed", "SL ranking did not complete")
    require(set(manifest["artifacts"]) == {"report.xlsx", "arrays.npz"}, "SL ranking artifact set differs")
    for name, checksum in manifest["artifacts"].items():
        require(hashlib.sha256((root / name).read_bytes()).hexdigest() == checksum,
                f"Artifact checksum mismatch: {name}")
    settings = manifest["settings"]
    parameters = {key: value for key, value in settings.items() if key not in ("folds", "tie_atol", "shifts")}
    expected_settings = ranking_settings(**parameters)
    has_diagnostics = schema != LEGACY_SCHEMA
    if has_diagnostics:
        expected_settings["shifts"] = settings["shifts"]
    require(settings == expected_settings, "SL ranking settings differ")
    names = [name for name, _ in candidates(settings["model_seed"])]
    require(manifest["candidates"] == names, "SL ranking candidate order differs")

    with np.load(root / "arrays.npz", allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
    target = arrays["target"]
    classes = np.arange(len(manifest["classes"]))
    require(target.ndim == 1 and np.issubdtype(target.dtype, np.integer)
            and np.array_equal(np.unique(target), classes), "Encoded target or class mapping differs")
    train, comparison, test = evaluation_split(target, classes, settings)
    assignments = cv_assignments(target[train], settings)
    plan = {"target": target, "train": train, "comparison": comparison, "test": test,
            "cv_fold_assignments": assignments}
    for name, expected in plan.items():
        require(np.issubdtype(arrays[name].dtype, np.integer) and np.array_equal(arrays[name], expected),
                f"SL ranking split or assignment differs: {name}")
    shapes = {
        "cv_predictions": (settings["cv_repeats"], len(names), len(train)),
        "test_predictions": (len(test),),
    }
    if len(comparison):
        shapes["comparison_predictions"] = (len(names), len(comparison))
    for name, shape in shapes.items():
        predictions = arrays[name]
        require(predictions.shape == shape and np.issubdtype(predictions.dtype, np.integer)
                and np.isin(predictions, classes).all(), f"Invalid SL ranking predictions: {name}")

    leader, expected_tables = cv_tables(
        target[train], classes, assignments, arrays["cv_predictions"], names, settings["metric"],
    )
    require((manifest["selected_candidate"], manifest["selected_model"]) == (leader, names[leader]),
            "CV leader differs")
    class_names = manifest["classes"] if schema == SCHEMA else None
    evidence = add_holdout_tables(expected_tables, target, classes, arrays, names, leader, settings, class_names=class_names)
    expected_arrays = set(plan) | set(shapes) | set(evidence)
    for name, expected in evidence.items():
        require(np.array_equal(arrays[name], expected), f"SL ranking bootstrap evidence differs: {name}")
    expected_tables.update(audit_tables(target, classes, arrays, names, leader, settings))
    if has_diagnostics:
        expected_arrays.update(validate_diagnostic_arrays(arrays, target[test], classes, manifest))
        expected_tables.update(ranking_uncertainty_tables(arrays, manifest))
    require(set(arrays) == expected_arrays, "SL ranking array set differs")
    text_columns = {**AUDIT_TEXT_COLUMNS, "Actual Class": str, "Class": str}
    actual_tables = pd.read_excel(root / "report.xlsx", sheet_name=None, keep_default_na=False, dtype=text_columns)
    require(set(actual_tables) == set(expected_tables), "SL ranking table names differ")
    require(manifest["sheets"] == {name: len(table) for name, table in expected_tables.items()},
            "SL ranking table row counts differ")
    for name, expected in expected_tables.items():
        same_table(actual_tables[name], expected, f"SL ranking table differs: {name}")
    if has_diagnostics:
        validate_metric_roundoff(actual_tables["Numerical_Metric"], expected_tables["Numerical_Metric"])
    return manifest
