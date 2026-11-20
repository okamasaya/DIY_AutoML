"""Compare a small H2O AutoML run against a saved DIY AutoML ranking."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import h2o
from h2o.automl import H2OAutoML
from h2o.backend import H2OLocalServer
import numpy as np
import pandas as pd

from diy_automl.data import load_csv
from diy_automl.metric_uncertainty import interval_rows
from diy_automl.reporting import fail, prepare, save
from diy_automl.sl_ranking import score_draws
from diy_automl.sl_ranking_validation import validate_sl_ranking
from diy_automl.supervised import METRICS, metrics


ALGORITHMS = ("GLM", "DRF", "GBM")
MAX_MODELS = 5


def fit_h2o(features, target, train, test, folds, classes, seed):
    # H2O treats ice_root as a URI and rejects spaces in the workspace path.
    with TemporaryDirectory(prefix="diy-automl-h2o-") as temporary:
        server = H2OLocalServer.start(
            nthreads=1, max_mem_size=1024**3, ice_root=temporary,
            bind_to_localhost=True, verbose=False,
            jvm_custom_args=["-Dfile.encoding=UTF-8", "-Dsun.stdout.encoding=UTF-8", "-Dsun.stderr.encoding=UTF-8",
                             "-Dsys.ai.h2o.telemetry.disabled=true"],
        )
        try:
            h2o.connect(server=server, verbose=False, telemetry=False)
            h2o.no_progress()
            train_labels = target[train]
            training = features.iloc[train].assign(label=train_labels, cv_fold=folds)
            frame = h2o.H2OFrame(training)
            frame["label"] = frame["label"].asfactor()
            if not np.array_equal(frame["cv_fold"].as_data_frame()["cv_fold"].to_numpy(), folds):
                raise ValueError("The uploaded fold assignments differ from the reference")
            automl = H2OAutoML(
                include_algos=list(ALGORITHMS), max_models=MAX_MODELS, seed=seed,
                sort_metric="logloss", keep_cross_validation_predictions=True,
            )
            automl.train(x=list(features.columns), y="label", training_frame=frame, fold_column="cv_fold")

            leaderboard = automl.leaderboard.as_data_frame()
            rows = []
            for entry in leaderboard.itertuples(index=False):
                model = h2o.get_model(entry.model_id)
                fold_column = model.actual_params.get("fold_column")
                if isinstance(fold_column, dict):
                    fold_column = fold_column.get("column_name")
                if fold_column != "cv_fold":
                    raise ValueError(f"The model did not use the supplied fold column: {entry.model_id}")
                holdout = model.cross_validation_holdout_predictions()
                if holdout is None:
                    raise ValueError(f"No cross-validation predictions for {entry.model_id}")
                predictions = holdout.as_data_frame()["predict"].to_numpy(dtype=int)
                fold_scores = [metrics(train_labels[part], predictions[part], classes)["F1"]
                               for part in (folds == fold for fold in np.unique(folds))]
                rows.append({"Model": entry.model_id, "Mean CV F1": float(np.mean(fold_scores)),
                             "CV Log Loss": entry.logloss})
            ranking = pd.DataFrame(rows).sort_values(["Mean CV F1", "CV Log Loss"], ascending=[False, True])
            leader = ranking.iloc[0]["Model"]
            held_out = h2o.H2OFrame(features.iloc[test])
            prediction = h2o.get_model(leader).predict(held_out).as_data_frame()["predict"].to_numpy(dtype=int)
            return ranking, leader, prediction
        finally:
            server.shutdown()


def compare(csv, reference, output):
    baseline = validate_sl_ranking(reference)
    settings = baseline["settings"]
    directory, manifest = prepare(output, 1, {
        "csv": str(csv), "reference": str(reference), "cv_repeat": 1,
        "algorithms": ALGORITHMS, "max_models": MAX_MODELS, "seed": settings["model_seed"],
    })
    manifest["dependencies"]["h2o"] = h2o.__version__
    try:
        if manifest["input_sha256"] != baseline["input_sha256"]:
            raise ValueError("The CSV does not match the saved DIY AutoML ranking")
        data = load_csv(csv, settings["target"])
        with np.load(Path(reference) / "arrays.npz", allow_pickle=False) as saved:
            train, test, target = (saved[key] for key in ("train", "test", "target"))
            folds = saved["cv_fold_assignments"][0]
            diy = saved["test_predictions"]
            reference_draws = saved["test_bootstrap_indices"]
        if not np.array_equal(data.y, target):
            raise ValueError("The encoded classes differ from the reference")
        features = pd.DataFrame(data.X, columns=[f"feature_{i}" for i in range(data.X.shape[1])])
        classes = np.unique(target)
        ranking, leader, prediction = fit_h2o(
            features, target, train, test, folds, classes, settings["model_seed"],
        )
        scores, samples, draws = score_draws(target[test], (diy, prediction), classes, settings)
        if not np.array_equal(draws, reference_draws):
            raise ValueError("The bootstrap rows differ from the reference")
        test_scores = pd.DataFrame(scores, columns=METRICS)
        test_scores.insert(0, "Workflow", ["DIY AutoML", "H2O AutoML"])
        differences = interval_rows(scores[1] - scores[0], samples[1] - samples[0], METRICS, settings["confidence"])
        manifest.update(selected_model=leader, diy_selected_model=baseline["selected_model"],
                        test_rows=len(test), train_rows=len(train))
        return save(directory, manifest, {
            "H2O_CV_Ranking": ranking, "Test_Scores": test_scores, "Test_Differences": differences,
        }, {
            "train": train, "test": test, "cv_folds": folds, "truth": target[test],
            "diy_predictions": diy, "h2o_predictions": prediction, "test_bootstrap_indices": draws,
        })
    except Exception as error:
        fail(directory, manifest, error)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    print(compare(arguments.csv, arguments.reference, arguments.output))
