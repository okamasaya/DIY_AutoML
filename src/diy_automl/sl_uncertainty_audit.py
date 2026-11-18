"""Audit one saved SL leader's sampling uncertainty and sensitivity to numerical inputs and offsets."""

from fractions import Fraction
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_csv
from .metric_uncertainty import interval_rows, score_samples, scores_from_confusion
from .reporting import fail, prepare, save, summary
from .supervised import METRICS, candidates, pipeline


SCHEMA = 11


def exact_scores(confusion):
    """Compute the same metrics with rational arithmetic on integer counts."""
    tp = [int(value) for value in np.diag(confusion)]
    observed = [int(value) for value in confusion.sum(axis=1)]
    predicted = [int(value) for value in confusion.sum(axis=0)]
    count = len(tp)
    return (
        Fraction(sum(tp), sum(observed)),
        sum(Fraction(hit, total or 1) for hit, total in zip(tp, predicted)) / count,
        sum(Fraction(hit, total or 1) for hit, total in zip(tp, observed)) / count,
        sum(Fraction(2 * hit, (actual + pred) or 1)
            for hit, actual, pred in zip(tp, observed, predicted)) / count,
    )


def parse_offsets(shifts, features):
    scenarios = []
    for shift in shifts:
        name, separator, number = shift.rpartition("=")
        if not separator or name not in features:
            raise ValueError(f"Use FEATURE=OFFSET with a CSV feature column: {shift!r}")
        try:
            offset = float(number)
        except ValueError as error:
            raise ValueError(f"Invalid offset: {shift!r}") from error
        if not np.isfinite(offset):
            raise ValueError(f"Offset must be finite: {shift!r}")
        scenarios.append({"feature": name, "offset": offset})
    return scenarios


def metric_rounding_table(confusion, points):
    rows = []
    for metric, point, exact in zip(METRICS, points, exact_scores(confusion)):
        error = float(Fraction.from_float(float(point)) - exact)
        rows.append({
            "Metric": metric, "Exact Fraction": f"{exact.numerator}/{exact.denominator}",
            "Float64 Score": point, "Signed Error": error, "Absolute Error": abs(error),
        })
    return pd.DataFrame(rows)


def confusion_counts(truth, predictions, class_count):
    encoded = truth * class_count + predictions
    return np.bincount(encoded, minlength=class_count ** 2).reshape(class_count, class_count)


def record_rounded_inputs(model, features, arrays):
    with np.errstate(over="ignore", under="ignore"):
        rounded = features.astype(np.float32).astype(np.float64)
    if np.isfinite(rounded).all():
        arrays["float32_input_predictions"] = model.predict(rounded)
        arrays["roundoff_by_feature"] = np.max(np.abs(rounded - features), axis=0)


def numerical_input_table(arrays, names, classes, points):
    if "float32_input_predictions" not in arrays:
        return pd.DataFrame([{"Status": "outside_float32_range"}])
    predictions = arrays["float32_input_predictions"]
    scores, _ = scores_from_confusion(confusion_counts(arrays["truth"], predictions, len(classes)))
    roundoff = arrays["roundoff_by_feature"]
    column = np.argmax(roundoff)
    return pd.DataFrame({
        "Status": "checked",
        "Metric": METRICS, "Float64 Input Score": points, "Float32 Rounded Input Score": scores,
        "Delta": scores - points, "Changed Predictions": np.count_nonzero(predictions != arrays["baseline_predictions"]),
        "Feature With Largest Roundoff": names[column],
        "Largest Input Roundoff": roundoff[column],
    })


def predict_offsets(model, features, names, scenarios):
    predictions = []
    for scenario in scenarios:
        changed = features.copy()
        column = names.index(scenario["feature"])
        with np.errstate(over="ignore"):
            changed[:, column] += scenario["offset"]
        if not np.isfinite(changed[:, column]).all():
            raise ValueError(f"Shift produced non-finite inputs: {scenario}")
        predictions.append(model.predict(changed))
    return np.stack(predictions)


def record_diagnostics(model, features, arrays, manifest):
    record_rounded_inputs(model, features, arrays)
    scenarios = manifest["settings"]["shifts"]
    if scenarios:
        arrays["offset_predictions"] = predict_offsets(model, features, manifest["features"], scenarios)


def offset_table(arrays, classes, scenarios, points, confidence):
    summaries = []
    for index, (scenario, prediction) in enumerate(zip(scenarios, arrays["offset_predictions"]), start=1):
        new_points, _, new_samples, _, _ = score_samples(
            arrays["truth"], prediction, classes, arrays["bootstrap_indices"],
        )
        table = interval_rows(new_points - points, new_samples - arrays["bootstrap_scores"], METRICS, confidence)
        table = table.rename(columns={"Estimate": "Delta"}).drop(columns="Status")
        table.insert(0, "Scenario", index)
        table.insert(1, "Feature", scenario["feature"])
        table.insert(2, "Assumed Offset", scenario["offset"])
        table["Baseline Score"] = points
        table["Shifted Score"] = new_points
        table["Changed Predictions"] = np.count_nonzero(prediction != arrays["baseline_predictions"])
        summaries.append(table)
    return pd.concat(summaries, ignore_index=True)


def diagnostic_tables(arrays, manifest):
    settings = manifest["settings"]
    classes = np.arange(len(manifest["classes"]))
    points, _ = scores_from_confusion(arrays["confusion"])
    tables = {
        "Numerical_Metric": metric_rounding_table(arrays["confusion"], points),
        "Numerical_Input": numerical_input_table(arrays, manifest["features"], classes, points),
    }
    if "roundoff_by_feature" in arrays:
        tables["Input_Roundoff"] = pd.DataFrame({
            "Feature": manifest["features"], "Maximum Absolute Roundoff": arrays["roundoff_by_feature"],
        })
    if settings["shifts"]:
        tables["Offset_Scenarios"] = offset_table(arrays, classes, settings["shifts"], points, settings["confidence"])
    return tables


def audit_tables(arrays, manifest):
    points, _ = scores_from_confusion(arrays["confusion"])
    return {
        "Summary": summary([
            ("Model", manifest["selected_model"]), ("Test Rows", len(arrays["test"])),
            ("Statistical", "Sampling uncertainty with fixed predictions and observed class counts"),
            ("Numerical Metric", "Float64 score versus the exact rational score from the same counts"),
            ("Numerical Input", "Rounding test inputs to Float32 and converting back to Float64 for prediction"),
            ("Systematic Scenarios", "Hypothetical constant offsets applied to test features"),
            ("Actual Systematic Bias", "Unmeasured. Calibration or independent labels are needed to estimate it."),
        ]),
        "Statistical": interval_rows(points, arrays["bootstrap_scores"], METRICS, manifest["settings"]["confidence"]),
        **diagnostic_tables(arrays, manifest),
    }


def audit_sl_uncertainty(source, csv, output, shift=()):
    """Run an audit of the fixed development-trained classifier from a completed rank-sl run."""
    from .sl_ranking import score_draws
    from .sl_ranking_validation import validate_sl_ranking

    source, csv, output = Path(source).resolve(), Path(csv).resolve(), Path(output).resolve()
    if output.is_relative_to(source):
        raise ValueError("Use a new output directory outside the saved ranking")
    original_manifest = source / "manifest.json"
    reference = validate_sl_ranking(source)
    input_hash = sha256(csv.read_bytes()).hexdigest()
    if input_hash != reference["input_sha256"]:
        raise ValueError("Input CSV SHA256 differs from the saved ranking")
    config = reference["settings"]
    data = load_csv(csv, config["target"])
    scenarios = parse_offsets(shift, data.feature_names)
    if data.classes != reference["classes"] or data.feature_names != reference["features"]:
        raise ValueError("CSV classes or features differ from the saved ranking")
    with np.load(source / "arrays.npz", allow_pickle=False) as stored:
        if not np.array_equal(stored["target"], data.y):
            raise ValueError("Encoded labels differ from the saved ranking")
        train, test, baseline = (stored[key] for key in ("train", "test", "test_predictions"))
        original_draws, original_samples = (stored[key] for key in ("test_bootstrap_indices", "test_bootstrap_scores"))
    settings = {
        "csv": str(csv), "target": config["target"], "source": str(source), "shifts": scenarios,
        **{name: config[name] for name in ("model_seed", "bootstrap_seed", "bootstrap_repeats", "confidence")},
    }
    output, manifest = prepare(output, SCHEMA, settings)
    manifest.update({
        "selected_model": reference["selected_model"], "classes": data.classes, "features": data.feature_names,
        "source_manifest_sha256": sha256(original_manifest.read_bytes()).hexdigest(),
        "source_artifacts": reference["artifacts"], "systematic_bias_status": "unmeasured",
    })
    try:
        if manifest["input_sha256"] != input_hash:
            raise ValueError("Input CSV changed during audit setup")
        model = pipeline(candidates(settings["model_seed"])[reference["selected_candidate"]][1])
        model.fit(data.X[train], data.y[train])
        test_features, truth = data.X[test], data.y[test]
        if not np.array_equal(model.predict(test_features), baseline):
            raise ValueError("Refitted model does not reproduce the saved test predictions")

        classes = np.arange(len(data.classes))
        _, samples, draws = score_draws(truth, baseline[None, :], classes, settings)
        samples = samples[0]
        if not np.array_equal(draws, original_draws) or not np.allclose(samples, original_samples, atol=1e-12, rtol=0):
            raise ValueError("Recalculated statistical evidence differs from the saved ranking")
        confusion = confusion_counts(truth, baseline, len(classes))
        arrays = {
            "train": train, "test": test, "truth": truth, "baseline_predictions": baseline,
            "confusion": confusion, "bootstrap_indices": draws, "bootstrap_scores": samples,
        }
        record_diagnostics(model, test_features, arrays, manifest)
        return save(output, manifest, audit_tables(arrays, manifest), arrays)
    except Exception as error:
        fail(output, manifest, error)
        raise
