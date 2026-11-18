"""Compare the fixed SL candidates on shared repeated folds and an optional comparison set."""

from numbers import Real

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from .clustering_subsample import integer
from .data import load_csv, split_indices
from .metric_uncertainty import (
    CLASS_METRICS, MAX_CLASSES, MAX_DRAWS, configuration, interval_rows, resample_indices, score_samples,
    scores_from_confusion,
)
from .reporting import fail, prepare, save, summary
from .sl_uncertainty_audit import confusion_counts, diagnostic_tables, parse_offsets, record_diagnostics
from .supervised import METRICS, candidates, metrics, pipeline
from .tuning import MAX_FEATURES, MAX_ROWS


LEGACY_SCHEMA = 10
UNCERTAINTY_SCHEMA = 12
SCHEMA = 13
FOLDS = 5
TIE_ATOL = 1e-12
CV_SCOPE = "Descriptive across repeated shared folds, with a fixed model seed. Fold scores are dependent."
TEST_SCOPE = "Reserved test for the CV-selected candidate only, conditional on the fitted model."
COMPARISON_SCOPE = (
    "Pointwise paired bootstrap vs the CV leader on held-out rows, "
    "conditional on fitted models and observed class counts. Assumes independent evaluation rows."
)


def ranking_settings(csv, target, seed, model_seed, cv_repeats, metric,
                     comparison_fraction, bootstrap_repeats, bootstrap_seed, confidence):
    if metric not in METRICS:
        raise ValueError("Unknown ranking metric")
    if (isinstance(comparison_fraction, bool) or not isinstance(comparison_fraction, Real)
            or not 0 <= comparison_fraction < 1):
        raise ValueError("comparison_fraction must be at least 0 and less than 1")
    bootstrap_repeats, confidence, bootstrap_seed, _ = configuration(
        bootstrap_repeats, confidence, bootstrap_seed,
    )
    return {
        "csv": str(csv), "target": target,
        "seed": integer(seed, "seed"), "model_seed": integer(model_seed, "model_seed"),
        "cv_repeats": integer(cv_repeats, "cv_repeats", 2, 20), "folds": FOLDS,
        "metric": metric, "comparison_fraction": float(comparison_fraction),
        "bootstrap_repeats": bootstrap_repeats, "bootstrap_seed": bootstrap_seed,
        "confidence": confidence, "tie_atol": TIE_ATOL,
    }


def evaluation_split(target, classes, settings):
    train, test = split_indices(target, settings["seed"])
    comparison = np.empty(0, dtype=train.dtype)
    fraction = settings["comparison_fraction"]
    if fraction:
        train, comparison = train_test_split(
            train, test_size=fraction, random_state=settings["seed"], stratify=target[train],
        )
    if np.bincount(target[train], minlength=len(classes)).min() < FOLDS:
        raise ValueError("Five rows per class are required in the CV training pool")
    for name, indices in (("test", test), ("comparison", comparison)):
        if len(indices) and not np.array_equal(np.unique(target[indices]), classes):
            raise ValueError(f"Every class must be represented in the {name} split")
        if len(indices) * settings["bootstrap_repeats"] > MAX_DRAWS:
            raise ValueError(f"Bootstrap row draw limit exceeded for the {name} split")
    return train, comparison, test


def cv_assignments(target, settings):
    splitter = RepeatedStratifiedKFold(
        n_splits=FOLDS, n_repeats=settings["cv_repeats"], random_state=settings["seed"],
    )
    assignments = np.empty((settings["cv_repeats"], len(target)), dtype=np.int8)
    for split, (_, validation) in enumerate(splitter.split(target[:, None], target)):
        repeat, fold = divmod(split, FOLDS)
        assignments[repeat, validation] = fold
    return assignments


def cv_predict(features, target, assignments, models):
    predictions = np.empty((len(assignments), len(models), len(target)), dtype=target.dtype)
    for repeat, folds in enumerate(assignments):
        for fold in range(FOLDS):
            train = folds != fold
            validation = folds == fold
            for candidate, (_, model) in enumerate(models):
                fitted = pipeline(clone(model)).fit(features[train], target[train])
                predictions[repeat, candidate, validation] = fitted.predict(features[validation])
    return predictions


def score_ranks(scores):
    """Competition ranks with ties within tolerance of the first score in each group."""
    order = np.argsort(-scores, kind="stable")
    ranks = np.ones(len(scores), dtype=int)
    rank = 1
    group_score = scores[order[0]]
    for position, candidate in enumerate(order[1:], start=2):
        if not np.isclose(scores[candidate], group_score, atol=TIE_ATOL, rtol=0):
            group_score = scores[candidate]
            rank = position
        ranks[candidate] = rank
    return ranks


def cv_tables(target, classes, assignments, predictions, names, metric):
    identity = [{"Candidate": i, "Model": name} for i, name in enumerate(names)]
    fold_rows = []
    for repeat, folds in enumerate(assignments):
        for fold in range(FOLDS):
            validation = folds == fold
            for candidate, model in enumerate(identity):
                scores = metrics(target[validation], predictions[repeat, candidate, validation], classes)
                fold_rows.append({"Repeat": repeat + 1, "Fold": fold + 1, **model, **scores})
    fold_table = pd.DataFrame(fold_rows)
    repeat_table = fold_table.groupby(["Repeat", "Candidate", "Model"], sort=False)[list(METRICS)].mean().reset_index()
    averages = repeat_table.groupby("Candidate", sort=False)[list(METRICS)].mean()
    overall_ranks = score_ranks(averages[metric].to_numpy())
    leader = int(np.flatnonzero(overall_ranks == 1)[0])
    repeat_scores = repeat_table[metric].to_numpy().reshape(len(assignments), len(names))
    repeat_ranks = np.stack([score_ranks(scores) for scores in repeat_scores])
    repeat_table["Rank"] = repeat_ranks.ravel()
    repeat_table["Score - CV Leader"] = (repeat_scores - repeat_scores[:, [leader]]).ravel()
    ranked = pd.DataFrame([
        {
            "Rank": overall_ranks[i], **model, **averages.iloc[i].to_dict(),
            "Repeat Score SD": repeat_scores[:, i].std(ddof=1),
            "Best Repeat Rank": repeat_ranks[:, i].min(),
            "Worst Repeat Rank": repeat_ranks[:, i].max(),
            "First Place Repeats": np.count_nonzero(repeat_ranks[:, i] == 1),
            "Score - CV Leader": averages[metric].iloc[i] - averages[metric].iloc[leader],
        }
        for i, model in enumerate(identity)
    ]).sort_values(["Rank", "Candidate"], ignore_index=True)
    return leader, {"CV_Ranking": ranked, "CV_Repeats": repeat_table, "CV_Folds": fold_table}


def score_draws(truth, predictions, classes, settings):
    indices = resample_indices(truth, settings["bootstrap_repeats"], settings["bootstrap_seed"])
    points, samples = [], []
    for prediction in predictions:
        point, _, sample, _, _ = score_samples(truth, prediction, classes, indices)
        points.append(point)
        samples.append(sample)
    return np.stack(points), np.stack(samples), indices


def classification_tables(confusion, class_names):
    labels = [f"'{label}" if label.startswith("=") else label for label in map(str, class_names)]
    matrix = pd.DataFrame(confusion, columns=[f"Predicted {label}" for label in labels])
    matrix.insert(0, "Actual Class", labels)

    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    correct = np.diag(confusion)
    _, scores = scores_from_confusion(confusion)
    by_class = pd.DataFrame({
        "Class": labels,
        "Support": support,
        "Predicted Count": predicted,
        "True Positive": correct,
        "False Positive": predicted - correct,
        "False Negative": support - correct,
    })
    by_class[list(CLASS_METRICS)] = scores
    return {"Test_Confusion": matrix, "Test_By_Class": by_class}


def add_holdout_tables(tables, target, classes, arrays, names, leader, settings, class_names=None):
    confidence = settings["confidence"]
    truth = target[arrays["test"]]
    predictions = arrays["test_predictions"]
    points, samples, draws = score_draws(
        truth, predictions[None, :], classes, settings,
    )
    tables.update({
        "Test_Score": pd.DataFrame([{"Model": names[leader], **dict(zip(METRICS, points[0]))}]),
        "Test_Intervals": interval_rows(points[0], samples[0], METRICS, confidence),
    })
    if class_names is not None:
        confusion = confusion_counts(truth, predictions, len(classes))
        tables.update(classification_tables(confusion, class_names))
    evidence = {"test_bootstrap_indices": draws, "test_bootstrap_scores": samples[0]}
    if len(arrays["comparison"]):
        points, samples, draws = score_draws(
            target[arrays["comparison"]], arrays["comparison_predictions"], classes, settings,
        )
        metric = METRICS.index(settings["metric"])
        differences = interval_rows(
            points[:, metric] - points[leader, metric],
            (samples[:, :, metric] - samples[leader, :, metric]).T, names, confidence,
        ).rename(columns={"Metric": "Model"}).drop(columns=["Status"])
        differences.insert(1, "CV Leader", names[leader])
        differences["Comparison Score"] = points[:, metric]
        conclusion = np.select(
            [~differences["Zero Width"] & (differences["Lower"] > 0),
             ~differences["Zero Width"] & (differences["Upper"] < 0)],
            ["above_reference", "below_reference"],
            default="direction_unresolved",
        )
        differences["Direction"] = conclusion
        differences.loc[leader, "Direction"] = "reference"
        tables["Comparison_Differences"] = differences
        ranking_columns = {
            "Estimate": "Comparison Score - CV Leader", "Lower": "Comparison Lower",
            "Upper": "Comparison Upper", "Direction": "Comparison Direction",
        }
        comparisons = differences.set_index("Model")[list(ranking_columns)].rename(columns=ranking_columns)
        tables["CV_Ranking"] = tables["CV_Ranking"].join(comparisons, on="Model")
        evidence.update(comparison_bootstrap_indices=draws, comparison_bootstrap_scores=samples)
    return evidence


def audit_tables(target, classes, arrays, names, leader, settings):
    counts = {"Encoded Class": classes}
    for key, column in (("train", "CV Rows"), ("comparison", "Comparison Rows"), ("test", "Test Rows")):
        counts[column] = np.bincount(target[arrays[key]], minlength=len(classes))
    return {
        "Summary": summary([
            ("CV Leader", names[leader]),
            ("Primary Metric", settings["metric"]),
            ("Ranking", "Mean across repeated 5-fold CV. Numerical ties use candidate order for selection."),
            ("First Place Repeats", "Counts each model in a first-place tie."),
            ("CV Scope", CV_SCOPE),
            ("Test Scope", TEST_SCOPE),
            ("Comparison Scope", COMPARISON_SCOPE if len(arrays["comparison"]) else "Comparison split not requested."),
            ("Comparison Use", "CV results determine the ranking and the final test candidate."),
        ]),
        "Split_Counts": pd.DataFrame(counts),
    }


def ranking_uncertainty_tables(arrays, manifest):
    truth = arrays["target"][arrays["test"]]
    baseline = arrays["test_predictions"]
    diagnostics = {
        **arrays, "truth": truth, "baseline_predictions": baseline,
        "bootstrap_indices": arrays["test_bootstrap_indices"],
        "bootstrap_scores": arrays["test_bootstrap_scores"],
        "confusion": confusion_counts(truth, baseline, len(manifest["classes"])),
    }
    scenarios = manifest["settings"]["shifts"]
    return {
        "Uncertainty_Summary": summary([
            ("Model", manifest["selected_model"]),
            ("Statistical", "Test_Intervals uses fixed predictions and fixed observed class counts."),
            ("Numerical Metric", "Float64 scores are compared with exact rational values from the same counts."),
            ("Numerical Input", "Test inputs are rounded through Float32 for prediction with the same model."),
            ("Systematic Scenarios", len(scenarios)),
            ("Actual Systematic Bias", "Unmeasured. Estimation requires independent calibration or labels."),
        ]),
        **diagnostic_tables(diagnostics, manifest),
    }


def rank_sl_csv(csv, target, output, seed=42, model_seed=42, cv_repeats=5, metric="F1",
                comparison_fraction=0.0, bootstrap_repeats=1000, bootstrap_seed=43, confidence=.95, shift=()):
    """Save a repeated CV ranking and the uncertainty evaluation of its test leader."""
    settings = ranking_settings(
        csv, target, seed, model_seed, cv_repeats, metric, comparison_fraction,
        bootstrap_repeats, bootstrap_seed, confidence,
    )
    data = load_csv(csv, target)
    settings["shifts"] = parse_offsets(shift, data.feature_names)
    classes = np.arange(len(data.classes))
    if len(data.y) > MAX_ROWS or data.X.shape[1] > MAX_FEATURES or len(classes) > MAX_CLASSES:
        raise ValueError(f"Ranking supports up to {MAX_ROWS} rows, {MAX_FEATURES} features and {MAX_CLASSES} classes")
    train, comparison, test = evaluation_split(data.y, classes, settings)
    models = candidates(settings["model_seed"])
    names = [name for name, _ in models]
    output, manifest = prepare(output, SCHEMA, settings)
    manifest.update(classes=data.classes, features=data.feature_names, candidates=names, stage="cv",
                    systematic_bias_status="unmeasured")
    try:
        assignments = cv_assignments(data.y[train], settings)
        predictions = cv_predict(data.X[train], data.y[train], assignments, models)
        leader, tables = cv_tables(data.y[train], classes, assignments, predictions, names, metric)
        arrays = {
            "target": data.y, "train": train, "test": test, "comparison": comparison,
            "cv_fold_assignments": assignments, "cv_predictions": predictions,
        }
        manifest.update(selected_candidate=leader, selected_model=names[leader], stage="holdout")
        fitted = pipeline(clone(models[leader][1])).fit(data.X[train], data.y[train])
        if len(comparison):
            comparison_predictions = []
            for i, (_, model) in enumerate(models):
                estimator = fitted if i == leader else pipeline(clone(model)).fit(data.X[train], data.y[train])
                comparison_predictions.append(estimator.predict(data.X[comparison]))
            arrays["comparison_predictions"] = np.stack(comparison_predictions)
        test_features = data.X[test]
        arrays["test_predictions"] = fitted.predict(test_features)
        evidence = add_holdout_tables(
            tables, data.y, classes, arrays, names, leader, settings, class_names=data.classes,
        )
        tables.update(audit_tables(data.y, classes, arrays, names, leader, settings))
        arrays.update(evidence)
        manifest["stage"] = "uncertainty"
        record_diagnostics(fitted, test_features, arrays, manifest)
        tables.update(ranking_uncertainty_tables(arrays, manifest))
        manifest["stage"] = "completed"
        return save(output, manifest, tables, arrays)
    except Exception as error:
        fail(output, manifest, error)
        raise
