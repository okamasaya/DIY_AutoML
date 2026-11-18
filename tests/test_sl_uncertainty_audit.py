from fractions import Fraction
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from diy_automl import audit_sl_uncertainty, rank_sl_csv
from diy_automl.data import load_csv
from diy_automl.sl_uncertainty_audit import exact_scores
from diy_automl.validation import validate


IRIS = Path(__file__).parents[1] / "data/reference/iris.csv"


def test_exact_macro_metrics_include_zero_denominators():
    assert exact_scores(np.array([[3, 1], [2, 6]])) == (
        Fraction(3, 4), Fraction(51, 70), Fraction(3, 4), Fraction(11, 15),
    )
    assert exact_scores(np.array([[2, 0], [2, 0]])) == (
        Fraction(1, 2), Fraction(1, 4), Fraction(1, 2), Fraction(1, 3),
    )


def test_audit_preserves_reference_and_uses_same_draws_for_offsets(tmp_path, monkeypatch):
    source = rank_sl_csv(IRIS, "target", tmp_path / "rank", cv_repeats=2, bootstrap_repeats=20)
    source_hashes = {path: sha256(path.read_bytes()).hexdigest() for path in source.iterdir()}
    data = load_csv(IRIS, "target")
    with np.load(source / "arrays.npz", allow_pickle=False) as stored:
        train = stored["train"]
        expected_draws = stored["test_bootstrap_indices"]
        baseline = stored["test_predictions"]
    original_fit = Pipeline.fit
    fits = []

    def fit(model, features, target, **kwargs):
        np.testing.assert_array_equal(features, data.X[train])
        np.testing.assert_array_equal(target, data.y[train])
        fits.append(model)
        return original_fit(model, features, target, **kwargs)

    monkeypatch.setattr(Pipeline, "fit", fit)
    output = audit_sl_uncertainty(source, IRIS, tmp_path / "audit",
                                  shift=("petal width (cm)=0", "petal width (cm)=0.5"))
    manifest = validate(output, 11)
    assert len(fits) == 1
    assert manifest["systematic_bias_status"] == "unmeasured"
    assert {path: sha256(path.read_bytes()).hexdigest() for path in source.iterdir()} == source_hashes

    results = pd.read_excel(output / "report.xlsx", sheet_name="Offset_Scenarios")
    labels = np.arange(len(data.classes))
    with np.load(output / "arrays.npz", allow_pickle=False) as evidence:
        np.testing.assert_array_equal(evidence["bootstrap_indices"], expected_draws)
        np.testing.assert_array_equal(evidence["offset_predictions"][0], baseline)
        shifted = evidence["offset_predictions"][1]
        truth = evidence["truth"]
        deltas = [
            f1_score(truth[rows], shifted[rows], labels=labels, average="macro")
            - f1_score(truth[rows], baseline[rows], labels=labels, average="macro")
            for rows in expected_draws
        ]
    rows = results.loc[results["Metric"].eq("F1")].set_index("Scenario")
    assert rows.loc[1, "Delta"] == 0
    assert rows.loc[2, "Changed Predictions"] == np.count_nonzero(shifted != baseline)
    np.testing.assert_allclose(rows.loc[2, ["Lower", "Upper"]].to_numpy(dtype=float),
                               np.quantile(deltas, [.025, .975]), atol=1e-12, rtol=0)
