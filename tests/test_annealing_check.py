import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from diy_automl import annealing_check_csv
from diy_automl.annealing import ANNEALING, BASELINE, EXHAUSTIVE, RANDOM, SubsetSpace, search_subsets
from diy_automl.annealing_workflow import SCHEMA
from diy_automl.data import split_indices
from diy_automl.validation import validate


IRIS = Path(__file__).parents[1] / "data/reference/iris.csv"


def test_annealing_rejects_downhill_moves_and_completes_the_budget():
    settings = {"search_seed": 44, "budget": 15, "initial_temperature": .01, "final_temperature": .0001}
    space = SubsetSpace(width=4, limit=4, budget=settings["budget"])
    first = search_subsets(space, RANDOM, settings, lambda mask: (0, 0))[0].mask
    history = search_subsets(space, ANNEALING, settings, lambda mask: (float(mask == first), 0))
    assert history[0].mask == first
    assert {trial.mask for trial in history} == set(space.exhaustive())
    assert len(history) == settings["budget"]
    assert history[0].temperature == pytest.approx(.01)
    assert history[-1].temperature == pytest.approx(.0001)
    assert any(trial.move == "neighbor" and not trial.accepted for trial in history)
    assert any(trial.move == "restart" for trial in history)
    assert history[-1].best == 0


def test_equal_fit_budget_and_no_test_leakage(tmp_path, monkeypatch):
    rows = np.arange(100)
    csv = tmp_path / "dataset.csv"
    y = rows % 2
    pd.DataFrame({"a": rows + .1, "b": rows + .2, "距離📏": rows + .3, "target": y}).to_csv(csv, index=False)
    train, test = split_indices(y)
    fit_rows = []
    original_fit = StandardScaler.fit
    original_predict = Pipeline.predict
    expected_cv_fits = 10 * (1 + 2 + 2 + 3)

    def fit(scaler, X, *args, **kwargs):
        fit_rows.append(set(np.floor(X[:, 0]).astype(int)))
        assert fit_rows[-1] <= set(train)
        return original_fit(scaler, X, *args, **kwargs)

    def predict(estimator, X, *args, **kwargs):
        if set(np.floor(X[:, 0]).astype(int)) == set(test):
            assert len(fit_rows) > expected_cv_fits
        return original_predict(estimator, X, *args, **kwargs)

    monkeypatch.setattr(StandardScaler, "fit", fit)
    monkeypatch.setattr(Pipeline, "predict", predict)
    result = annealing_check_csv(csv, "target", tmp_path / "audit", budget=2, max_features=1, exhaustive=True,
                                 bootstrap_repeats=8, shift=("a=0.1", "b=0.1", "距離📏=0.1"))
    assert len(fit_rows) == expected_cv_fits + 4
    assert fit_rows[-4:] == [set(train)] * 4
    manifest = validate(result, SCHEMA)
    assert len(fit_rows) == expected_cv_fits + 4
    assert manifest["methods"] == [BASELINE, ANNEALING, RANDOM, EXHAUSTIVE]
    assert manifest["features"] == ["a", "b", "距離📏"]
    tables = pd.read_excel(result / "report.xlsx", sheet_name=None)
    assert tables["Selection"]["CV Fits"].tolist() == [10, 20, 20, 30]
    scenario = tables["Offset_Scenarios"]
    assert (~scenario["Feature Used"]).any()
    assert scenario.loc[~scenario["Feature Used"], "Changed Predictions"].eq(0).all()


def test_exhaustive_check_and_paired_interval_from_saved_predictions(tmp_path):
    output = annealing_check_csv(IRIS, "target", tmp_path / "audit", budget=8, exhaustive=True, bootstrap_repeats=16)
    manifest = validate(output, SCHEMA)
    sheets = pd.read_excel(output / "report.xlsx", sheet_name=None)
    history = sheets["Search_History"]
    exhaustive = history[history["Method"] == EXHAUSTIVE]
    assert len(exhaustive) == 15
    assert sheets["Selection"]["CV Score"].iloc[-1] == exhaustive["CV Score"].max()

    artifact = output / "arrays.npz"
    with np.load(artifact, allow_pickle=False) as saved:
        arrays = dict(saved)
    truth = arrays["target"][arrays["test"]]
    _, annealing, random, _ = arrays["test_predictions"]
    differences = [
        f1_score(truth[draw], annealing[draw], average="macro") - f1_score(truth[draw], random[draw], average="macro")
        for draw in arrays["test_bootstrap_indices"]
    ]
    table = sheets["Test_Differences"]
    saved_interval = table.loc[
        (table["Method"] == ANNEALING) & (table["Reference"] == RANDOM) & (table["Metric"] == "F1"),
        ["Lower", "Upper"],
    ].iloc[0]
    np.testing.assert_allclose(saved_interval, np.quantile(differences, [.025, .975]), atol=1e-12, rtol=0)

    arrays["cv_masks"][1, 0] = ~arrays["cv_masks"][1, 0]
    np.savez_compressed(artifact, **arrays)
    manifest["artifacts"]["arrays.npz"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="CV proposal differs"):
        validate(output, SCHEMA)
