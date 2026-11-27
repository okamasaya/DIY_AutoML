"""Compare four decision-tree depths on one development split."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from diy_automl.data import dl_indices, load_csv
from diy_automl.reporting import fail, file_sha256, prepare, save, summary
from diy_automl.supervised import metrics, pipeline


SCHEMA = "decision-tree-depth/v1"
DEPTHS = (1, 3, 5, None)


def compare_depths(csv, target, output, *, split_seed=42, model_seed=42, score_holdout=False):
    input_hash = file_sha256(csv)
    dataset = load_csv(csv, target)
    train, validation, reserved = dl_indices(dataset.y, split_seed)
    train_features, train_truth = dataset.X[train], dataset.y[train]
    validation_features, validation_truth = dataset.X[validation], dataset.y[validation]
    classes = np.arange(len(dataset.classes))
    settings = {
        "csv": str(csv), "target": target, "split_seed": split_seed, "model_seed": model_seed,
        "max_depths": DEPTHS, "cpu_threads": 1, "score_holdout": score_holdout,
    }
    output, manifest = prepare(output, SCHEMA, settings)

    try:
        if manifest["input_sha256"] != input_hash:
            raise ValueError("The CSV changed between reading and recording the input")
        rows, train_predictions, validation_predictions = [], [], []
        estimators = [pipeline(DecisionTreeClassifier(max_depth=depth, random_state=model_seed)) for depth in DEPTHS]
        with threadpool_limits(limits=1):
            for depth, estimator in zip(DEPTHS, estimators, strict=True):
                start = perf_counter()
                estimator.fit(train_features, train_truth)
                seconds = perf_counter() - start
                train_prediction = estimator.predict(train_features)
                validation_prediction = estimator.predict(validation_features)
                train_f1 = metrics(train_truth, train_prediction, classes)["F1"]
                validation_f1 = metrics(validation_truth, validation_prediction, classes)["F1"]
                rows.append({
                    "Max depth": depth if depth is not None else "Unlimited",
                    "Actual depth": estimator[-1].get_depth(),
                    "Train macro F1": train_f1,
                    "Validation macro F1": validation_f1,
                    "Train minus validation F1": train_f1 - validation_f1,
                    "Fit seconds": seconds,
                })
                train_predictions.append(train_prediction)
                validation_predictions.append(validation_prediction)

        comparison = pd.DataFrame(rows)
        comparison["Validation F1 minus unrestricted"] = (
            comparison["Validation macro F1"] - rows[-1]["Validation macro F1"]
        )
        selected_index = int(comparison["Validation macro F1"].idxmax())
        arrays = {
            "train": train, "validation": validation, "reserved": reserved,
            "train_truth": train_truth, "validation_truth": validation_truth,
            "train_predictions": np.stack(train_predictions),
            "validation_predictions": np.stack(validation_predictions),
        }
        if score_holdout:
            holdout_truth = dataset.y[reserved]
            with threadpool_limits(limits=1):
                holdout_predictions = np.stack([estimator.predict(dataset.X[reserved]) for estimator in estimators])
            comparison["Holdout macro F1"] = [metrics(holdout_truth, pred, classes)["F1"] for pred in holdout_predictions]
            comparison["Holdout F1 minus unrestricted"] = (
                comparison["Holdout macro F1"] - comparison["Holdout macro F1"].iloc[-1]
            )
            arrays.update(holdout_truth=holdout_truth, holdout_predictions=holdout_predictions)

        manifest.update(
            experiment_sha256=file_sha256(Path(__file__)),
            classes=dataset.classes,
            features=dataset.feature_names,
            selection={"criterion": "validation macro F1", "selected_max_depth": DEPTHS[selected_index]},
            partitions={"train": len(train), "validation": len(validation), "reserved": len(reserved)},
        )
        sheets = {
            "Depth_Comparison": comparison,
            "Summary": summary([
                ("Protocol", "One stratified development split"),
                ("Train rows", len(train)),
                ("Validation rows", len(validation)),
                ("Reserved rows", len(reserved)),
                ("Selection", "Highest validation macro F1; prefer the lower depth limit on ties"),
                ("Selected max depth", rows[selected_index]["Max depth"]),
                ("Holdout", "Scored after selection" if score_holdout else "Reserved without scoring"),
            ]),
        }
        return save(output, manifest, sheets, arrays)
    except Exception as error:
        fail(output, manifest, error)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--score-holdout", action="store_true", help="Score all depths after fixing the validation choice")
    print(compare_depths(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
