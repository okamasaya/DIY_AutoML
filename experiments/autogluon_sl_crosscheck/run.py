"""Compare AutoGluon Tabular and DIY AutoML on a saved rank-sl partition."""

import argparse
import hashlib
import importlib.metadata
from pathlib import Path
import platform
from tempfile import TemporaryDirectory

from autogluon.tabular import TabularPredictor
import numpy as np
import pandas as pd

from diy_automl.data import load_csv
from diy_automl.metric_uncertainty import interval_rows
from diy_automl.reporting import fail, save, write_json
from diy_automl.sl_ranking import FOLDS, add_holdout_tables, audit_tables, cv_tables, score_draws
from diy_automl.sl_ranking_validation import validate_sl_ranking
from diy_automl.supervised import METRICS, metrics


BASE_MODELS = ("RF", "XT", "KNN", "LR")
NAMES = (*BASE_MODELS, "WeightedEnsemble_L2")
AG_METRICS = {"F1": "f1_macro", "Accuracy": "accuracy", "Precision": "precision_macro", "Recall": "recall_macro"}
FIT_OPTIONS = {
    "fit_weighted_ensemble": True, "num_bag_folds": 0, "num_stack_levels": 0,
    "num_cpus": 1, "num_gpus": 0, "calibrate_decision_threshold": False, "raise_on_model_failure": True,
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit(training, label, classes, path, settings):
    seed = settings["model_seed"]
    parameters = {key: {"ag_args": {"name": key}} for key in BASE_MODELS}
    for key in ("RF", "XT"):
        parameters[key].update(n_estimators=100, random_state=seed)
    parameters["LR"]["random_state"] = seed
    predictor = TabularPredictor(
        label=label, eval_metric=AG_METRICS[settings["metric"]], path=str(path), verbosity=0,
        problem_type="binary" if len(classes) == 2 else "multiclass",
        learner_kwargs={"random_state": seed},
    )
    predictor.fit(training, hyperparameters=parameters, **FIT_OPTIONS)
    refitted = predictor.refit_full(model="all", set_best_to_refit_full=False, num_cpus=1, num_gpus=0)
    if set(refitted) != set(NAMES):
        raise ValueError(f"Expected every AutoGluon candidate to refit, got {refitted}")
    return predictor, refitted


def predict_checked(predictor, model, features, truth, classes, metric, checks):
    predicted = predictor.predict(features, model=model)
    prediction = predicted.to_numpy()
    if not predicted.index.equals(features.index) or not np.isin(prediction, classes).all():
        raise ValueError("AutoGluon predictions do not match evaluation rows or classes")
    observed = pd.Series(truth, index=features.index)
    result = predictor.evaluate_predictions(observed, predicted, auxiliary_metrics=False, silent=True)
    difference = abs(result[AG_METRICS[metric]] - metrics(truth, prediction, classes)[metric])
    if not np.isfinite(difference) or difference > 1e-12:
        raise ValueError(f"AutoGluon metric differs for {model}: {difference}")
    checks.append(difference)
    return prediction


def crosscheck(csv, reference, output):
    reference, output = reference.resolve(), output.resolve()
    if output.is_relative_to(reference):
        raise ValueError("Use an output directory outside the saved DIY ranking")
    diy = validate_sl_ranking(reference)
    input_hash = sha256(csv)
    if input_hash != diy["input_sha256"]:
        raise ValueError("Input CSV SHA256 differs from the saved DIY ranking")
    settings = diy["settings"]
    label = settings["target"]
    metric = settings["metric"]
    data = load_csv(csv, label)
    classes = np.arange(len(data.classes))
    features = pd.DataFrame(data.X, columns=data.feature_names)
    labeled = features.assign(**{label: data.y})
    with np.load(reference / "arrays.npz", allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in ("target", "train", "test", "comparison", "cv_fold_assignments")}
        diy_test_pred = stored["test_predictions"]
        diy_test_draws = stored["test_bootstrap_indices"]
        diy_test_scores = stored["test_bootstrap_scores"]
    if not np.array_equal(data.y, arrays["target"]) or data.classes != diy["classes"] or sha256(csv) != input_hash:
        raise ValueError("Input CSV or its encoded class order differs from the saved DIY ranking")
    train, test, comparison = (arrays[key] for key in ("train", "test", "comparison"))
    assignments = arrays["cv_fold_assignments"]
    diy_rank = pd.read_excel(reference / "report.xlsx", sheet_name="CV_Ranking").iloc[0]
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running", "stage": "cv", "protocol": "autogluon_sl_crosscheck",
        "input_csv": str(csv.resolve()), "input_sha256": input_hash,
        "reference": str(reference), "reference_manifest_sha256": sha256(reference / "manifest.json"),
        "reference_artifacts": diy["artifacts"], "script_sha256": sha256(Path(__file__)),
        "python": platform.python_version(), "platform": platform.platform(),
        "dependencies": {name: importlib.metadata.version(name) for name in
                         ("autogluon.tabular", "numpy", "pandas", "scipy", "scikit-learn", "openpyxl")},
        "settings": settings, "fit_options": FIT_OPTIONS, "candidates": NAMES,
        "classes": data.classes, "features": data.feature_names,
    }
    write_json(output / "manifest.json", manifest)
    checks = []
    try:
        cv_pred = np.empty((len(assignments), len(NAMES), len(train)), dtype=data.y.dtype)
        for repeat, folds in enumerate(assignments):
            for fold in range(FOLDS):
                mask = folds == fold
                rows = train[mask]
                with TemporaryDirectory(prefix="diy-ag-cv-") as temporary:
                    predictor, refitted = fit(labeled.iloc[train[~mask]], label, classes,
                                              Path(temporary) / "predictor", settings)
                    for i, name in enumerate(NAMES):
                        cv_pred[repeat, i, mask] = predict_checked(
                            predictor, refitted[name], features.iloc[rows], data.y[rows], classes, metric, checks,
                        )
            print(f"AutoGluon CV repeat {repeat + 1}/{len(assignments)} completed", flush=True)
        leader, tables = cv_tables(data.y[train], classes, assignments, cv_pred, NAMES, metric)
        selected = NAMES[leader]
        manifest.update(stage="refit", selected_model=selected, diy_selected_model=diy["selected_model"])
        predictor, refitted = fit(labeled.iloc[train], label, classes, output / "predictor", settings)
        manifest["selected_refit"] = refitted[selected]
        if len(comparison):
            arrays["comparison_predictions"] = np.stack([
                predict_checked(predictor, refitted[name], features.iloc[comparison],
                                data.y[comparison], classes, metric, checks)
                for name in NAMES
            ])
        arrays.update(
            cv_predictions=cv_pred, diy_test_predictions=diy_test_pred,
            test_predictions=predict_checked(
                predictor, refitted[selected], features.iloc[test], data.y[test], classes, metric, checks,
            ),
        )
        manifest["stage"] = "evaluation"
        arrays.update(add_holdout_tables(tables, data.y, classes, arrays, NAMES, leader, settings))
        tables.update(audit_tables(data.y, classes, arrays, NAMES, leader, settings))

        points, samples, draws = score_draws(
            data.y[test], np.stack([diy_test_pred, arrays["test_predictions"]]), classes, settings,
        )
        if not np.array_equal(draws, diy_test_draws) or not np.allclose(samples[0], diy_test_scores, atol=1e-12, rtol=0):
            raise ValueError("Saved DIY final-test bootstrap differs on the shared draws")
        arrays["crosscheck_test_scores"] = samples
        delta = interval_rows(points[1] - points[0], samples[1] - samples[0], METRICS, settings["confidence"])
        delta = delta.rename(columns={"Estimate": "AutoGluon - DIY"})
        delta["DIY Test Score"] = points[0]
        delta["AutoGluon Test Score"] = points[1]
        ag_rank = tables["CV_Ranking"].iloc[0]
        index = METRICS.index(metric)
        leaders = pd.DataFrame([
            {"Workflow": "DIY AutoML", "CV Leader": diy["selected_model"],
             "Primary Metric": metric, "Mean CV Score": diy_rank[metric], "Test Score": points[0, index]},
            {"Workflow": "AutoGluon", "CV Leader": selected,
             "Primary Metric": metric, "Mean CV Score": ag_rank[metric], "Test Score": points[1, index]},
        ])
        manifest.update(stage="completed", metric_checks=len(checks), max_metric_difference=max(checks))
        save(output, manifest, {"Crosscheck_Leaders": leaders, "Crosscheck_Test_Delta": delta, **tables}, arrays)
        print(output)
        print(leaders.to_string(index=False))
        print(delta.loc[delta["Metric"] == metric, ["Metric", "AutoGluon - DIY", "Lower", "Upper"]].to_string(index=False))
    except Exception as error:
        fail(output, manifest, error)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True, help="A completed rank-sl output directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    crosscheck(args.csv, args.reference, args.output)
