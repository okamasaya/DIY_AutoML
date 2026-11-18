import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from diy_automl import rank_sl_csv
from diy_automl.sl_ranking import SCHEMA, classification_tables, evaluation_split, ranking_settings, score_ranks
from diy_automl.validation import validate


IRIS = Path(__file__).parents[1] / "data/reference/iris.csv"


def test_ranking_ties_share_position():
    np.testing.assert_array_equal(score_ranks(np.array([.8, .7, .8 + 5e-13, .65])), [1, 3, 1, 4])


def test_class_report_keeps_an_unpredicted_class():
    tables = classification_tables(np.array([[3, 0], [2, 0]]), ["common", "rare"])
    row = tables["Test_By_Class"].set_index("Class").loc["rare"]
    assert row["Support"] == row["False Negative"] == 2
    assert row[["Predicted Count", "Precision", "Recall", "F1"]].eq(0).all()


def test_comparison_and_test_rows_stay_out_of_fits(tmp_path, monkeypatch):
    count = 100
    target = np.arange(count) % 2
    data = pd.DataFrame({"row": np.arange(count), "signal": target, "target": target})
    csv = tmp_path / "dataset.csv"
    data.to_csv(csv, index=False)
    settings = ranking_settings(csv, "target", 42, 42, 2, "F1", .25, 5, 43, .95)
    train, comparison, test = evaluation_split(target, np.arange(2), settings)
    fits, predictions, test_models = [], [], []
    original_fit, original_predict = StandardScaler.fit, Pipeline.predict

    def fit(scaler, features, *args, **kwargs):
        rows = set(features[:, 0])
        assert rows <= set(train)
        fits.append(rows)
        return original_fit(scaler, features, *args, **kwargs)

    def predict(model, features, *args, **kwargs):
        rows = set(features[:, 0])
        predictions.append(rows)
        if rows == set(test):
            test_models.append(model)
        return original_predict(model, features, *args, **kwargs)

    monkeypatch.setattr(StandardScaler, "fit", fit)
    monkeypatch.setattr(Pipeline, "predict", predict)
    output = rank_sl_csv(csv, "target", tmp_path / "run", cv_repeats=2,
                         comparison_fraction=.25, bootstrap_repeats=5, shift=("signal=0.5",))
    assert len(fits) == 2 * 5 * 5 + 5
    assert predictions.count(set(comparison)) == 5
    assert len(test_models) == 3  # Original inputs, Float32 round trip and the requested offset.
    assert all(model is test_models[0] for model in test_models)
    validate(output, SCHEMA)
    assert len(fits) == 2 * 5 * 5 + 5


def test_saved_cv_predictions_and_paired_differences(tmp_path):
    output = rank_sl_csv(IRIS, "target", tmp_path / "rank", cv_repeats=2,
                         comparison_fraction=.25, bootstrap_repeats=20)
    manifest = validate(output, SCHEMA)
    sheets = pd.read_excel(output / "report.xlsx", sheet_name=None)
    leader = manifest["selected_candidate"]
    names = manifest["candidates"]
    assert sheets["CV_Ranking"]["Model"].iloc[0] == names[leader]
    assert sheets["Test_Score"]["Model"].tolist() == [names[leader]]
    assert {"Test_Intervals", "Uncertainty_Summary", "Numerical_Metric", "Numerical_Input"} <= set(sheets)
    assert "Offset_Scenarios" not in sheets
    assert manifest["systematic_bias_status"] == "unmeasured"

    artifact = output / "arrays.npz"
    with np.load(artifact, allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
    target = arrays["target"]
    labels = np.arange(len(manifest["classes"]))
    test_truth, test_predictions = target[arrays["test"]], arrays["test_predictions"]
    matrix = confusion_matrix(test_truth, test_predictions, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        test_truth, test_predictions, labels=labels, zero_division=0,
    )
    np.testing.assert_array_equal(sheets["Test_Confusion"].iloc[:, 1:].to_numpy(), matrix)
    by_class = sheets["Test_By_Class"]
    assert by_class["Class"].tolist() == manifest["classes"]
    np.testing.assert_array_equal(by_class["Support"], support)
    np.testing.assert_array_equal(by_class["Predicted Count"], matrix.sum(axis=0))
    np.testing.assert_allclose(by_class[["Precision", "Recall", "F1"]],
                               np.column_stack([precision, recall, f1]), atol=1e-12, rtol=0)
    cv_truth = target[arrays["train"]]
    for repeat, assignments in enumerate(arrays["cv_fold_assignments"]):
        scores = [
            np.mean([f1_score(cv_truth[assignments == fold], predictions[assignments == fold],
                              labels=labels, average="macro", zero_division=0) for fold in range(5)])
            for predictions in arrays["cv_predictions"][repeat]
        ]
        rows = sheets["CV_Repeats"].query("Repeat == @repeat + 1")
        np.testing.assert_allclose(rows["F1"], scores, atol=1e-12, rtol=0)

    truth = target[arrays["comparison"]]
    scores = arrays["comparison_bootstrap_scores"]
    predictions = arrays["comparison_predictions"]
    for draw_no, draw in enumerate(arrays["comparison_bootstrap_indices"]):
        independent = [f1_score(truth[draw], pred[draw], labels=labels, average="macro", zero_division=0)
                       for pred in predictions]
        np.testing.assert_allclose(scores[:, draw_no, 3], independent, atol=1e-12, rtol=0)
    deltas = scores[:, :, 3] - scores[leader, :, 3]
    np.testing.assert_allclose(
        sheets["Comparison_Differences"][["Lower", "Upper"]],
        np.quantile(deltas, [.025, .975], axis=1).T, atol=1e-12, rtol=0,
    )
    draws = arrays["comparison_bootstrap_indices"]
    draws[0, 0] = (draws[0, 0] + 1) % len(truth)
    np.savez_compressed(artifact, **arrays)
    manifest["artifacts"]["arrays.npz"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="bootstrap evidence differs"):
        validate(output, SCHEMA)
