import json
from pathlib import Path

import numpy as np
import pandas as pd

from diy_automl import render_excel_report, select_csv, tune_csv
from diy_automl.excel_report_validation import validate_excel_report
from diy_automl.run_progress import PROGRESS_LOG
from diy_automl.validation import validate


IRIS = Path(__file__).parents[1] / "data/reference/iris.csv"


def test_selection_saves_progress_and_keeps_it_in_illustrated_results(tmp_path, capsys):
    output = select_csv(IRIS, "target", tmp_path / "select", bootstrap_repeats=3, progress="both")
    manifest = validate(output, 3)
    assert manifest["settings"]["progress"] == "both"
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "candidate 5/5 | fold 5/5" in captured.err
    progress = pd.read_excel(output / "report.xlsx", sheet_name="Progress")
    scores = pd.read_excel(output / "report.xlsx", sheet_name="SL_CV_Folds")
    folds = progress.loc[progress.stage == "cv"].reset_index(drop=True)
    assert folds["model"].tolist() == scores["Model"].tolist()
    assert folds["fold"].tolist() == (scores["Fold"] + 1).tolist()
    pd.testing.assert_series_equal(folds["score"], scores["F1"], check_names=False)
    assert progress.stage.tolist()[-4:] == ["refit", "holdout", "bootstrap", "report"]

    illustrated = render_excel_report(output, tmp_path / "illustrated", charts="detail")
    report = validate_excel_report(illustrated)
    assert [figure["id"] for figure in report["figures"]] == [
        "sl_selected_cm", "sl_training_cv", "sl_fold_scores", "sl_cv_progress",
    ]
    with np.load(illustrated / "figure_data.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["sl_fold_scores__scores"], scores["F1"].to_numpy().reshape(5, 5))
        np.testing.assert_array_equal(data["sl_cv_progress__completed_folds"], np.arange(26))
    assert (illustrated / PROGRESS_LOG).read_bytes() == (output / PROGRESS_LOG).read_bytes()


def test_tune_excel_only_records_each_search_fold(tmp_path, capsys):
    output = tune_csv(IRIS, "target", tmp_path / "tune", bootstrap_repeats=3, tree_max_depth=3, progress="excel")
    validate(output, 7)
    assert capsys.readouterr().err == ""
    rows = [json.loads(line) for line in (output / PROGRESS_LOG).read_text(encoding="utf-8").splitlines()]
    cv = [event for event in rows if event["stage"] == "cv"]
    assert [(event["candidate"], event["fold"]) for event in cv] == [
        (candidate, fold) for candidate in range(1, 22) for fold in range(1, 6)
    ]
    assert all(event["candidates"] == 21 and event["folds"] == 5 for event in cv)
