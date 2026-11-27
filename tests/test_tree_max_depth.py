import json

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_info

from diy_automl import annealing_check_csv, audit_sl_uncertainty, rank_sl_csv, run_csv, select_csv, tune_csv
from diy_automl.validation import validate


@pytest.fixture
def csv(tmp_path):
    rows = np.arange(80)
    target = rows % 2
    path = tmp_path / "separable.csv"
    pd.DataFrame({"signal": target, "row": rows, "target": target}).to_csv(path, index=False)
    return path


@pytest.mark.parametrize("workflow,options", [
    (run_csv, {"methods": ("sl",)}),
    (select_csv, {"bootstrap_repeats": 3}),
    (rank_sl_csv, {"cv_repeats": 2, "bootstrap_repeats": 3}),
    (tune_csv, {"bootstrap_repeats": 3}),
    (annealing_check_csv, {"model": "Decision Tree", "budget": 2, "bootstrap_repeats": 3}),
])
@pytest.mark.parametrize("depth_options,forest_depth", [
    ({"tree_max_depth": 1}, 1),
    ({"tree_max_depth": 2, "tree_max_depths": {"Decision Tree": 1}}, 2),
])
def test_depth_is_used_in_fits_and_saved(csv, tmp_path, monkeypatch, workflow, options, depth_options, forest_depth):
    fitted_types = set()
    original_fit = Pipeline.fit
    compute_options = {"cpu_threads": 1}
    if workflow is not annealing_check_csv:
        compute_options["forest_jobs"] = 2

    def fit(pipeline, features, target, **kwargs):
        model = pipeline[-1]
        if not fitted_types:
            assert all(pool["num_threads"] <= 1 for pool in threadpool_info())
        if isinstance(model, (DecisionTreeClassifier, RandomForestClassifier)):
            assert model.max_depth == (1 if isinstance(model, DecisionTreeClassifier) else forest_depth)
            fitted_types.add(type(model))
        if isinstance(model, RandomForestClassifier):
            assert model.n_jobs == 2
        return original_fit(pipeline, features, target, **kwargs)

    monkeypatch.setattr(Pipeline, "fit", fit)
    output = workflow(csv, "target", tmp_path / "run", **depth_options, **compute_options, **options)
    manifest = validate(output)
    expected_types = {DecisionTreeClassifier}
    if workflow is not annealing_check_csv:
        expected_types.add(RandomForestClassifier)
    assert fitted_types == expected_types
    assert all(manifest["settings"][key] == value for key, value in (depth_options | compute_options).items())
    report = pd.read_excel(output / "report.xlsx", sheet_name="Summary").set_index("Item")
    assert str(report.loc["Tree max depth", "Value"]) == str(depth_options["tree_max_depth"])
    assert str(report.loc["Native CPU threads", "Value"]) == "1"
    if "tree_max_depths" in depth_options:
        assert str(report.loc["Decision Tree max depth", "Value"]) == "1"
    if workflow is tune_csv:
        assert manifest["sheets"]["SL_Search_Settings"] == 21


def test_uncertainty_audit_refits_saved_depth(csv, tmp_path, monkeypatch):
    depths = {"Decision Tree": 1}
    source = rank_sl_csv(csv, "target", tmp_path / "rank", tree_max_depth=2, tree_max_depths=depths,
                         cpu_threads=1, forest_jobs=2, cv_repeats=2, bootstrap_repeats=3)
    assert validate(source)["selected_model"] == "Decision Tree"
    original_fit = Pipeline.fit
    fitted_depths = []

    def fit(pipeline, features, target, **kwargs):
        assert all(pool["num_threads"] <= 1 for pool in threadpool_info())
        fitted_depths.append(pipeline[-1].max_depth)
        return original_fit(pipeline, features, target, **kwargs)

    monkeypatch.setattr(Pipeline, "fit", fit)
    output = audit_sl_uncertainty(source, csv, tmp_path / "audit")
    settings = validate(output)["settings"]
    assert settings["tree_max_depths"] == depths
    assert (settings["cpu_threads"], settings["forest_jobs"]) == (1, 2)
    assert fitted_depths == [1]


def test_tune_keeps_depth_search_for_unspecified_model(csv, tmp_path):
    output = tune_csv(csv, "target", tmp_path / "tune", tree_max_depths={"Decision Tree": 2}, bootstrap_repeats=3)
    assert validate(output)["sheets"]["SL_Search_Settings"] == 23
    configurations = pd.read_excel(output / "report.xlsx", sheet_name="SL_Search_Settings")
    for model, expected in (("Decision Tree", {2}), ("Random Forest", {None, 5})):
        params = configurations.loc[configurations.Model == model, "Estimator Parameters"]
        assert {json.loads(item)["max_depth"] for item in params} == expected


def test_annealing_rejects_depth_for_another_model(csv, tmp_path):
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="must target the selected"):
        annealing_check_csv(csv, "target", output, model="Decision Tree", tree_max_depths={"Random Forest": 3})
    assert not output.exists()


def test_cpu_thread_limit_is_restored_on_failure():
    from diy_automl.sl_compute import limit_sl_threads

    before = threadpool_info()
    with pytest.raises(RuntimeError, match="simulated failure"):
        with limit_sl_threads({"cpu_threads": 1}):
            assert all(pool["num_threads"] <= 1 for pool in threadpool_info())
            raise RuntimeError("simulated failure")
    assert threadpool_info() == before
