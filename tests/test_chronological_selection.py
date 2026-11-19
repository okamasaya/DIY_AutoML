from datetime import date, timedelta
from random import Random
import csv

import numpy as np
import pandas as pd
from openpyxl import load_workbook
import pytest

from diy_automl.bp_evaluation import evaluate_selection
from diy_automl.bp_preview import projection_features, run_projection, save_audit
from diy_automl.calendar_comparison import BY_MONTH, BY_MONTH_CLASS, calendar_comparison, calendar_predictions
from diy_automl.chronological_selection import chronological_partitions, require_class_coverage
from diy_automl.data import load_csv
from diy_automl.excel_report import render_excel_report
from diy_automl.excel_report_validation import validate_excel_report
from diy_automl.reporting import file_sha256, write_json
from diy_automl.uncertainty_workflow import evaluate_uncertainty
from diy_automl.validation import validate


def test_bp_chronological_selection_keeps_entire_dates_in_order(tmp_path):
    source = tmp_path / "observations.csv"
    start = date(2024, 1, 14)
    rows = []
    for offset in range(20):
        day = start + timedelta(days=offset)
        following_day = day + timedelta(days=1)
        rows.extend([(f"{day}T12:00Z", offset, "A"), (f"{following_day}T01:00+02:00", offset + 30, "B")])
    Random(42).shuffle(rows)
    with source.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["記録日時", "値", "結果"])
        writer.writerows(rows)

    original = save_audit(source, tmp_path / "original", "結果", time_column="記録日時")
    assert projection_features(original.profile) == ["値"]
    selection = run_projection(original, ["値"])
    assert selection.audit.profile["time"] == original.profile["time"]
    output, result = evaluate_selection(selection, tmp_path / "evaluation")
    manifest = validate(output, 3)
    assert manifest["features"] == ["値"]
    assert manifest["split"]["clock"] == "utc"
    assert manifest["split"]["test"] == {"start": "2024-01-30", "end": "2024-02-02", "days": 4, "rows": 8}
    rule = {"uses_hours": True, "source_rows": {"weekday_hour": 8, "weekday": 0, "overall": 0}}
    assert manifest["calendar_rule"] == rule
    assert result["calendar"] == {**rule, "correct": 8, "f1": 1.0, "majority_f1": 1 / 3}
    assert result["f1_lower"] is None and result["f1_upper"] is None
    assert pd.read_excel(output / "report.xlsx", sheet_name="SL_Bootstrap").empty
    assert pd.read_excel(output / "report.xlsx", sheet_name="SL_Test_Intervals").empty
    workbook = load_workbook(output / "bp_report.xlsx", read_only=True)
    try:
        summary = dict(workbook["まず読む"].values)
        assert "2024-01-30〜2024-02-02" in summary["比較した範囲"]
        assert "未算出" in summary["統計的なばらつき"]
        assert "曜日・時間帯の多数決" in [row[0] for row in workbook["単純ルール比較"].values]
        assert {row[0] for row in list(workbook["月別"].values)[1:]} == {"2024-01", "2024-02"}
        assert len(list(workbook["月別・ラベル別"].values)) == 13
    finally:
        workbook.close()

    with pytest.raises(ValueError, match="Row bootstrap is unavailable"):
        evaluate_uncertainty(output, tmp_path / "uncertainty")

    rendered = render_excel_report(output, tmp_path / "illustrated", csv=selection.csv)
    assert "bp_report.xlsx" in validate_excel_report(rendered)["source_artifacts"]

    with np.load(output / "arrays.npz", allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    days, train, test = (arrays[key] for key in ("observation_days", "train", "test"))
    assert days[train].max() < days[test].min()
    np.testing.assert_array_equal(arrays["calendar_predictions"], arrays["truth"])
    np.testing.assert_array_equal(np.unique(arrays["observation_hours"]), [12, 23])
    np.testing.assert_array_equal(np.unique(days, return_counts=True)[1], np.full(20, 2))
    for i in range(5):
        fit, review = (arrays[f"fold_{i}_{role}"] for role in ("train", "validation"))
        assert days[fit].max() < days[review].min()
        assert days[review].max() < days[test].min()

    original_prediction = arrays["calendar_predictions"][0]
    arrays["calendar_predictions"][0] = 1 - original_prediction
    np.savez_compressed(output / "arrays.npz", **arrays)
    manifest["artifacts"]["arrays.npz"] = file_sha256(output / "arrays.npz")
    write_json(output / "manifest.json", manifest)
    with pytest.raises(ValueError, match="Calendar comparison predictions differ"):
        validate(output, 3)
    arrays["calendar_predictions"][0] = original_prediction
    arrays["fold_0_train"][0] = arrays["fold_0_validation"][0]
    np.savez_compressed(output / "arrays.npz", **arrays)
    manifest["artifacts"]["arrays.npz"] = file_sha256(output / "arrays.npz")
    write_json(output / "manifest.json", manifest)
    with pytest.raises(ValueError, match="Invalid chronological CV indices"):
        validate(output, 3)


@pytest.mark.parametrize(("dates", "match"), [
    (["", "2025-03-10"], "Missing timestamp"),
    (["10/03/2025", "2025-03-10"], "Expected an ISO 8601 timestamp"),
    (["2025-03-09T23:30Z", "2025-03-10"], "timestamps consistently"),
])
def test_chronological_selection_rejects_invalid_dates(tmp_path, dates, match):
    source = tmp_path / "dates.csv"
    source.write_text(f"day,value,label\n{dates[0]},1,A\n{dates[1]},2,B\n", encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_csv(source, "label", time_column="day")


def test_chronological_selection_does_not_move_boundaries_to_fit_labels():
    days = np.repeat(np.arange(20), 2)
    train, test, folds = chronological_partitions(days)
    labels = np.tile([0, 1], 20)
    labels[test] = 0
    with pytest.raises(ValueError, match="holdout period must contain every class"):
        require_class_coverage(labels, train, test, folds)
    with pytest.raises(ValueError, match="at least six training dates"):
        chronological_partitions(np.repeat(np.arange(7), 2))


def test_calendar_rule_uses_training_calendar_majorities_with_fallbacks(tmp_path):
    source = tmp_path / "mixed-precision.csv"
    source.write_text("day,value,label\n2024-01-01T08:00,1,B\n2024-01-01,2,B\n2024-01-02,3,A\n"
                      "2024-01-02T12:00,4,A\n2024-01-02T12:40,5,B\n2024-01-02,6,A\n", encoding="utf-8")
    training = load_csv(source, "label", time_column="day")
    np.testing.assert_array_equal(training.observation_hours, [8, -1, -1, 12, 12, -1])
    monday = date(2024, 1, 8).toordinal()
    predictions, overall, rule = calendar_predictions(
        training.observation_days, training.observation_hours, training.y,
        np.array([monday, monday, monday, monday + 1, monday + 2]), np.array([8, 17, -1, 12, 8]),
    )
    np.testing.assert_array_equal(predictions, [1, 1, 1, 0, 0])
    np.testing.assert_array_equal(overall, [0] * 5)
    assert rule == {"uses_hours": True, "source_rows": {"weekday_hour": 2, "weekday": 2, "overall": 1}}


def test_calendar_report_notes_absent_classes_and_split_months_in_date_only_input():
    jan = date(2024, 1, 1).toordinal()
    feb = date(2024, 2, 1).toordinal()
    days = np.array([jan, jan + 1, jan + 28, feb, feb + 1])
    arrays = {
        "train": np.array([0, 1]), "test": np.array([2, 3, 4]), "observation_days": days,
        "observation_hours": np.full(len(days), -1), "train_truth": np.array([0, 1]),
        "truth": np.array([0, 1, 0]), "predictions": np.array([0, 1, 1]),
    }
    tables, _, rule = calendar_comparison(arrays, class_count=2)
    assert rule == {"uses_hours": False, "source_rows": {"weekday_hour": 0, "weekday": 1, "overall": 2}}
    monthly = tables[BY_MONTH].query("Method == 'Selected model'")
    assert monthly["First date"].tolist() == ["2024-01-29", "2024-02-01"]
    assert monthly["Last date"].tolist() == ["2024-01-29", "2024-02-02"]
    assert monthly["F1"].tolist()[0] == 0.5
    missing = tables[BY_MONTH_CLASS].query("Month == '2024-01' and `Class index` == 1")
    assert (missing["Support"] == 0).all() and missing["Recall"].isna().all()
