"""Compare feature subset searches on shared folds, then evaluate their fixed winners."""

from numbers import Real
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.base import clone

from .annealing import ANNEALING, BASELINE, EXHAUSTIVE, RANDOM, SubsetSpace, columns, search_subsets
from .clustering_subsample import integer
from .data import load_csv
from .metric_uncertainty import MAX_CLASSES, interval_rows, scores_from_confusion
from .reporting import fail, prepare, save, summary
from .sl_compute import limit_sl_threads, sl_compute_summary
from .sl_ranking import classification_tables, cv_assignments, cv_predict, evaluation_split, ranking_settings, score_draws
from .sl_uncertainty_audit import confusion_counts, diagnostic_tables, parse_offsets, predict_offsets, record_rounded_inputs
from .supervised import METRICS, TREE_MODELS, candidates, pipeline, sl_model_options, tree_depth_options, tree_depth_summary
from .tuning import MAX_FEATURES, MAX_ROWS


SCHEMA = 14
MAX_BUDGET = 128
MAX_EXHAUSTIVE = 255
MAX_CV_FITS = 4000
MAX_OFFSETS = 20
DIAGNOSTICS = ("float32_input_predictions", "roundoff_by_feature", "offset_predictions")


def annealing_settings(csv, target, width, seed, model, model_seed, cv_repeats, metric,
                       budget, max_features, search_seed, initial_temperature, final_temperature,
                       exhaustive, bootstrap_repeats, bootstrap_seed, confidence, shifts,
                       *, tree_max_depth=None, tree_max_depths=None, cpu_threads=None, forest_jobs=None):
    settings = ranking_settings(
        csv, target, seed, model_seed, cv_repeats, metric, 0, bootstrap_repeats, bootstrap_seed, confidence,
        tree_max_depth=tree_max_depth, tree_max_depths=tree_max_depths,
        cpu_threads=cpu_threads, forest_jobs=forest_jobs,
    )
    models = dict(candidates(settings["model_seed"]))
    if model not in models:
        raise ValueError(f"Unknown classifier: {model!r}")
    if tree_depth_options(settings) and (model not in TREE_MODELS or set(settings.get("tree_max_depths", {})) - {model}):
        raise ValueError("Tree depth settings must target the selected Decision Tree or Random Forest")
    if forest_jobs is not None and model != "Random Forest":
        raise ValueError("forest_jobs requires choosing Random Forest")
    limit = width if max_features is None else integer(max_features, "max_features", 1, width)
    budget = integer(budget, "budget", 1, MAX_BUDGET)
    temperatures = initial_temperature, final_temperature
    if (any(isinstance(t, bool) or not isinstance(t, Real) or not np.isfinite(t) or t <= 0
            for t in temperatures) or final_temperature > initial_temperature):
        raise ValueError("Temperatures must be finite, positive and non-increasing")
    if not isinstance(exhaustive, bool):
        raise ValueError("exhaustive must be a bool")
    if len(shifts) > MAX_OFFSETS:
        raise ValueError(f"At most {MAX_OFFSETS} offset scenarios are supported")
    settings.update(
        model=model, budget=budget, max_features=limit, search_seed=integer(search_seed, "search_seed"),
        initial_temperature=float(initial_temperature), final_temperature=float(final_temperature),
        exhaustive=exhaustive, shifts=shifts,
    )
    return settings


def search_plan(width, settings):
    space = SubsetSpace(width, settings["max_features"], settings["budget"])
    methods = [BASELINE, ANNEALING, RANDOM]
    subset_count = 1 + 2 * min(space.size, settings["budget"])
    if settings["exhaustive"]:
        if space.size > MAX_EXHAUSTIVE:
            raise ValueError(f"Exhaustive search supports at most {MAX_EXHAUSTIVE} subsets (got {space.size})")
        methods.append(EXHAUSTIVE)
        subset_count += space.size
    if subset_count * settings["folds"] * settings["cv_repeats"] > MAX_CV_FITS:
        raise ValueError(f"The plan exceeds {MAX_CV_FITS} cross-validation fits")
    return space, methods


def subset_score(truth, classes, assignments, predictions, settings):
    values = []
    for folds, predicted in zip(assignments, predictions):
        for fold in range(settings["folds"]):
            held = folds == fold
            matrix = confusion_counts(truth[held], predicted[held], len(classes))
            values.append(scores_from_confusion(matrix)[0])
    return float(np.mean(values, axis=0)[METRICS.index(settings["metric"])])


def search_tables(histories, manifest):
    settings, features = manifest["settings"], manifest["features"]
    folds = settings["folds"] * settings["cv_repeats"]
    selections, selected_features, trace = [], [], []
    anchors = {method: trials[trials[-1].best] for method, trials in histories.items()}
    for method, trials in histories.items():
        winner = anchors[method]
        subset = columns(winner.mask, len(features))
        row = {
            "Method": method, "Best Step": trials[-1].best + 1, "Features": len(subset),
            "CV Score": winner.score, "CV Delta vs All": winner.score - anchors[BASELINE].score,
            "Subsets Scored": len(trials), "CV Fits": len(trials) * folds,
            "CV Time (s)": sum(trial.seconds for trial in trials),
        }
        if EXHAUSTIVE in anchors:
            row["CV Gap to Exhaustive"] = anchors[EXHAUSTIVE].score - winner.score if method != BASELINE else None
        selections.append(row)
        for index in subset:
            selected_features.append({"Method": method, "Column Position": index + 1, "Feature": features[index]})
        for step, trial in enumerate(trials, start=1):
            trace.append({
                "Method": method, "Step": step, "Proposal": "".join(
                    "1" if trial.mask & (1 << index) else "0" for index in range(len(features))
                ),
                "Move": trial.move, "Features": trial.mask.bit_count(), "Temperature": trial.temperature,
                "CV Score": trial.score, "Accepted": trial.accepted, "Current Step": trial.current + 1,
                "Best Step": trial.best + 1, "CV Fits": folds, "CV Time (s)": trial.seconds,
            })
    return {
        "Selection": pd.DataFrame(selections),
        "Selected_Features": pd.DataFrame(selected_features),
        "Search_History": pd.DataFrame(trace),
    }


def holdout_tables(arrays, manifest, histories):
    settings, features = manifest["settings"], manifest["features"]
    truth = arrays["target"][arrays["test"]]
    classes = np.arange(len(manifest["classes"]))
    predictions = arrays["test_predictions"]
    methods = list(histories)
    points, samples, draws = score_draws(truth, predictions, classes, settings)
    tables = {
        "Test_Score": pd.DataFrame(points, columns=METRICS).assign(
            Method=methods, **{"Refit Time (s)": arrays["test_seconds"]}
        )[["Method", *METRICS, "Refit Time (s)"]],
    }
    differences = []
    for candidate, reference in [(i, 0) for i in range(1, len(methods))] + [(1, 2)]:
        table = interval_rows(
            points[candidate] - points[reference], samples[candidate] - samples[reference],
            METRICS, settings["confidence"],
        )
        table.insert(0, "Reference", methods[reference])
        table.insert(0, "Method", methods[candidate])
        differences.append(table)
    tables["Test_Differences"] = pd.concat(differences, ignore_index=True)

    by_method = {}
    for index, (method, history) in enumerate(histories.items()):
        names = [features[i] for i in columns(history[history[-1].best].mask, len(features))]
        prefix = f"test_{index}_"
        diagnostics = {key: arrays[prefix + key] for key in DIAGNOSTICS if prefix + key in arrays}
        matrix = confusion_counts(truth, predictions[index], len(classes))
        diagnostics.update(
            truth=truth, confusion=matrix, baseline_predictions=predictions[index],
            bootstrap_indices=draws, bootstrap_scores=samples[index],
        )
        method_tables = {
            "Test_Intervals": interval_rows(points[index], samples[index], METRICS, settings["confidence"]),
            **classification_tables(matrix, manifest["classes"]),
            **diagnostic_tables(diagnostics, {**manifest, "features": names}),
        }
        if settings["shifts"]:
            scenarios = method_tables["Offset_Scenarios"]
            scenarios["Feature Used"] = scenarios["Feature"].isin(names)
        for name, frame in method_tables.items():
            frame.insert(0, "Method", method)
            by_method.setdefault(name, []).append(frame)
    tables.update({name: pd.concat(frames, ignore_index=True) for name, frames in by_method.items()})
    return tables, {"test_bootstrap_indices": draws, "test_bootstrap_scores": samples}


def report_tables(histories, arrays, manifest, space):
    settings = manifest["settings"]
    tables = search_tables(histories, manifest)
    holdout, evidence = holdout_tables(arrays, manifest, histories)
    tables.update(holdout)
    tables["Summary"] = summary([
        ("Classifier", settings["model"]),
        ("CV Metric", settings["metric"]),
        ("Search Space Size", str(space.size)),
        ("CV Fits Total", int(tables["Selection"]["CV Fits"].sum())),
        ("Holdout Refits", len(histories)),
        ("CV", "Shared stratified folds. The scaler is fitted inside each training fold."),
        ("Selection", "Highest mean fold score rounded to 12 decimals, then fewer columns, then lower bit mask."),
        ("Proposal", "Mask bits follow original CSV feature order from left to right."),
        ("Test", "One fixed winner per arm. Differences use paired, stratified row bootstrap, conditional on fitted models."),
        ("Systematic Bias", "Unmeasured. Offsets are hypothetical test scenarios."),
        *tree_depth_summary(settings),
        *sl_compute_summary(settings),
    ])
    for frame in tables.values():
        for name in ("Feature", "Feature With Largest Roundoff"):
            if name in frame:
                frame[name] = frame[name].map(lambda text: "'" + text if isinstance(text, str) and text.startswith("=") else text)
    return tables, evidence


def record_holdout(data, train, test, model, histories, settings, arrays):
    predictions, seconds = [], []
    for index, history in enumerate(histories.values()):
        subset = columns(history[history[-1].best].mask, data.X.shape[1])
        names = [data.feature_names[i] for i in subset]
        start = perf_counter()
        fitted = pipeline(clone(model)).fit(data.X[np.ix_(train, subset)], data.y[train])
        seconds.append(perf_counter() - start)
        features = data.X[np.ix_(test, subset)]
        prediction = fitted.predict(features)
        predictions.append(prediction)

        diagnostics = {}
        record_rounded_inputs(fitted, features, diagnostics)
        shifts = settings["shifts"]
        if shifts:
            active = [i for i, shift in enumerate(shifts) if shift["feature"] in names]
            offsets = np.repeat(prediction[None, :], len(shifts), axis=0)
            if active:
                offsets[active] = predict_offsets(fitted, features, names, [shifts[i] for i in active])
            diagnostics["offset_predictions"] = offsets
        arrays.update({f"test_{index}_{key}": value for key, value in diagnostics.items()})
    arrays.update(test_predictions=np.stack(predictions), test_seconds=np.asarray(seconds))


def annealing_check_csv(csv, target, output, seed=42, model="SVM", model_seed=42, cv_repeats=2, metric="F1",
                        budget=32, max_features=None, search_seed=44, initial_temperature=.03,
                        final_temperature=.002, exhaustive=False, bootstrap_repeats=1000,
                        bootstrap_seed=43, confidence=.95, shift=(), *, tree_max_depth=None, tree_max_depths=None,
                        cpu_threads=None, forest_jobs=None):
    """Compare classical annealing and uniform random subset search with an equal CV fit budget."""
    data = load_csv(csv, target)
    width = data.X.shape[1]
    classes = np.arange(len(data.classes))
    if len(data.y) > MAX_ROWS or width > MAX_FEATURES or len(classes) > MAX_CLASSES:
        raise ValueError(f"Annealing check supports up to {MAX_ROWS} rows, {MAX_FEATURES} features and {MAX_CLASSES} classes")
    settings = annealing_settings(
        csv, target, width, seed, model, model_seed, cv_repeats, metric, budget, max_features,
        search_seed, initial_temperature, final_temperature, exhaustive, bootstrap_repeats,
        bootstrap_seed, confidence, parse_offsets(shift, data.feature_names),
        tree_max_depth=tree_max_depth, tree_max_depths=tree_max_depths,
        cpu_threads=cpu_threads, forest_jobs=forest_jobs,
    )
    space, methods = search_plan(width, settings)
    train, _, test = evaluation_split(data.y, classes, settings)
    training, truth = data.X[train], data.y[train]
    assignments = cv_assignments(truth, settings)
    estimator = dict(candidates(settings["model_seed"], **sl_model_options(settings)))[model]
    output, manifest = prepare(output, SCHEMA, settings)
    manifest.update(classes=data.classes, features=data.feature_names, methods=methods, stage="cv")
    try:
        masks, predictions = [], []

        def evaluate(mask):
            subset = columns(mask, width)
            start = perf_counter()
            predicted = cv_predict(training[:, subset], truth, assignments, [(model, estimator)])[:, 0, :]
            elapsed = perf_counter() - start
            masks.append([bool(mask & (1 << index)) for index in range(width)])
            predictions.append(predicted.astype(np.uint8))
            return subset_score(truth, classes, assignments, predicted, settings), elapsed

        with limit_sl_threads(settings):
            histories = {method: search_subsets(space, method, settings, evaluate) for method in methods}
        arrays = {
            "target": data.y, "train": train, "test": test, "cv_fold_assignments": assignments,
            "cv_masks": np.asarray(masks), "cv_predictions": np.stack(predictions),
            "cv_seconds": np.array([trial.seconds for history in histories.values() for trial in history]),
        }
        manifest["stage"] = "holdout"
        with limit_sl_threads(settings):
            record_holdout(data, train, test, estimator, histories, settings, arrays)
        tables, evidence = report_tables(histories, arrays, manifest, space)
        arrays.update(evidence)
        manifest["stage"] = "completed"
        return save(output, manifest, tables, arrays)
    except Exception as error:
        fail(output, manifest, error)
        raise
