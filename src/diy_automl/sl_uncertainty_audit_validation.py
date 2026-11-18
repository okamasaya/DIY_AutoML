"""Verify saved SL audit calculations without refitting or reading the source CSV."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metric_uncertainty import observations
from .sl_ranking import score_draws
from .sl_uncertainty_audit import SCHEMA, audit_tables, confusion_counts, parse_offsets
from .ul_subsample_validation import require, same_table


AUDIT_TEXT_COLUMNS = {"Feature": str, "Feature With Largest Roundoff": str}


def validate_diagnostic_arrays(arrays, truth, classes, manifest):
    require(manifest["systematic_bias_status"] == "unmeasured", "Unexpected systematic bias status")
    shifts = manifest["settings"]["shifts"]
    tokens = [f"{item['feature']}={item['offset']}" for item in shifts]
    require(parse_offsets(tokens, manifest["features"]) == shifts, "Unexpected offset scenarios")
    expected = set()
    if "float32_input_predictions" in arrays:
        observations(truth, arrays["float32_input_predictions"], classes)
        roundoff = arrays["roundoff_by_feature"]
        require(roundoff.shape == (len(manifest["features"]),)
                and np.isfinite(roundoff).all() and (roundoff >= 0).all(), "Invalid input roundoff")
        expected.update(("float32_input_predictions", "roundoff_by_feature"))
    if shifts:
        predictions = arrays["offset_predictions"]
        require(predictions.shape == (len(shifts), len(truth)), "Offset predictions have unexpected shape")
        for row in predictions:
            observations(truth, row, classes)
        expected.add("offset_predictions")
    return expected


def validate_metric_roundoff(actual, expected):
    columns = ["Signed Error", "Absolute Error"]
    require(np.allclose(actual[columns], expected[columns], atol=0, rtol=1e-12), "Metric rounding error differs")


def validate_sl_uncertainty_audit(directory):
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text())
    require(manifest["schema_version"] == SCHEMA and manifest["status"] == "completed",
            "Uncertainty audit did not complete")
    require(set(manifest["artifacts"]) == {"report.xlsx", "arrays.npz"}, "Unexpected audit artifact set")
    for name, digest in manifest["artifacts"].items():
        require(hashlib.sha256((root / name).read_bytes()).hexdigest() == digest,
                f"Audit artifact checksum differs: {name}")

    settings = manifest["settings"]
    with np.load(root / "arrays.npz", allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
    classes = np.arange(len(manifest["classes"]))
    truth, baseline, _ = observations(arrays["truth"], arrays["baseline_predictions"], classes)
    train, test = arrays["train"], arrays["test"]
    indices = np.concatenate([train, test])
    require(all(rows.ndim == 1 and np.issubdtype(rows.dtype, np.integer) for rows in (train, test))
            and len(train) > 0 and len(test) == len(truth)
            and (indices >= 0).all() and len(np.unique(indices)) == len(indices), "Invalid audit splits")
    require(np.array_equal(arrays["confusion"], confusion_counts(truth, baseline, len(classes))),
            "Audit confusion counts differ")
    _, samples, draws = score_draws(truth, baseline[None, :], classes, settings)
    require(np.array_equal(arrays["bootstrap_indices"], draws)
            and np.array_equal(arrays["bootstrap_scores"], samples[0]), "Audit bootstrap evidence differs")

    expected_arrays = {"truth", "baseline_predictions", "train", "test", "confusion",
                       "bootstrap_indices", "bootstrap_scores"}
    expected_arrays.update(validate_diagnostic_arrays(arrays, truth, classes, manifest))
    require(set(arrays) == expected_arrays, "Unexpected audit arrays")
    expected = audit_tables(arrays, manifest)
    actual = pd.read_excel(
        root / "report.xlsx", sheet_name=None, keep_default_na=False,
        dtype=AUDIT_TEXT_COLUMNS,
    )
    require(set(actual) == set(expected) and manifest["sheets"] == {key: len(frame) for key, frame in expected.items()},
            "Unexpected audit tables")
    for name, frame in expected.items():
        same_table(actual[name], frame, f"Audit table differs: {name}")
    validate_metric_roundoff(actual["Numerical_Metric"], expected["Numerical_Metric"])
    return manifest
