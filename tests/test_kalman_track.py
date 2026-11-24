import json

import numpy as np
import pandas as pd
import pytest

from diy_automl import kalman_track_csv
from diy_automl.kalman import filter_series
from diy_automl.reporting import file_sha256, write_json
from diy_automl.validation import validate


def test_irregular_steps_skip_missing_update_and_use_previous_input():
    times = [0, 1, 3, 4]
    obs = [0, np.nan, 8, 13]
    control = [2, 1, 6, 99]
    options = dict(observation_sd=1, process_noise=0.3, initial_rate=1, initial_rate_sd=2)
    trace = filter_series(times, obs, control, **options)
    np.testing.assert_allclose(trace["forecast"][1:3], [[2, 3], [10, 5]])
    np.testing.assert_allclose(trace["forecast_covariance"][1], [[5.1, 4.15], [4.15, 4.3]])
    np.testing.assert_array_equal(trace["filtered"][1], trace["forecast"][1])
    np.testing.assert_array_equal(trace["filtered_covariance"][1], trace["forecast_covariance"][1])

    prior = trace["forecast_covariance"][2]
    expected = prior - np.outer(prior[:, 0], prior[0]) / (prior[0, 0] + 1)
    np.testing.assert_allclose(trace["filtered_covariance"][2], expected, atol=1e-12)
    assert np.linalg.eigvalsh(trace["filtered_covariance"]).min() >= 0

    altered = filter_series(times, [0, np.nan, -80, 13], [2, 1, -60, 99], **options)
    np.testing.assert_allclose(trace["forecast"][:3], altered["forecast"][:3], equal_nan=True)
    assert trace["filtered"][2, 0] != altered["filtered"][2, 0]
    np.testing.assert_allclose(trace["dynamics"][1:, 0], [2, 10, 18])
    assert trace["persistence"][2] == 0 and trace["persistence"][3] == 8


def test_csv_report_distinguishes_state_from_observation_and_validates(tmp_path):
    source = tmp_path / "trajectory.csv"
    source.write_text("time,observed,reference\n0,,0\n1,1,1\n2,,2\n4,4,4\n5,5,5\n", encoding="utf-8")
    result = kalman_track_csv(source, tmp_path / "run", time_column="time", observation_column="observed",
                              reference_column="reference", observation_sd=0.5, process_noise=0.2)
    manifest = validate(result, 18)
    assert manifest["input_sha256"] == file_sha256(source)
    sheet = pd.read_excel(result / "report.xlsx", sheet_name="Track")
    assert sheet["Forecast value"].iloc[:2].isna().all()
    assert np.isnan(sheet.loc[0, "Filtered value"])
    assert sheet.loc[1, "Filtered value"] == 1
    assert sheet.loc[2, "Forecast origin"] == 1
    assert sheet.loc[2, "Forecast value"] == sheet.loc[2, "Filtered value"]
    assert sheet.loc[3, "Forecast observation upper"] > sheet.loc[3, "Forecast state upper"]
    metrics = pd.read_excel(result / "report.xlsx", sheet_name="Metrics").set_index(["Stage", "Target", "Method"])
    assert metrics.loc[("Forecast", "Observation", "Kalman"), "Rows"] == 2
    assert metrics.loc[("Forecast", "Reference state", "Kalman"), "Rows"] == 3
    assert metrics.loc[("Filtered", "Reference state", "Raw observation"), "Rows"] == 2

    # Recalculation should detect altered arrays even if their checksum was replaced.
    artifact = result / "arrays.npz"
    with np.load(artifact, allow_pickle=False) as stored:
        arrays = dict(stored)
    arrays["forecast"][2, 0] += 1
    np.savez_compressed(artifact, **arrays)
    manifest["artifacts"][artifact.name] = file_sha256(artifact)
    write_json(result / "manifest.json", manifest)
    with pytest.raises(ValueError, match="Tracking array differs: forecast"):
        validate(result, 18)


@pytest.mark.parametrize(("rows", "message"), [
    ("0,1\n0,2\n", "strictly increasing"),
    ("0,NaN\n1,2\n", "Nonfinite observation"),
    ("0,\n1,\n", "At least one observation"),
    ("0,1\n1,2,3\n", "Unexpected column count"),
])
def test_bad_series_is_rejected_before_creating_output(tmp_path, rows, message):
    source = tmp_path / "input.csv"
    source.write_text("time,value\n" + rows, encoding="utf-8")
    output = tmp_path / "result"
    with pytest.raises(ValueError, match=message):
        kalman_track_csv(source, output, time_column="time", observation_column="value",
                         observation_sd=1, process_noise=0)
    assert not output.exists()


def test_no_reference_and_only_last_row_observed(tmp_path):
    source = tmp_path / "series.csv"
    source.write_text("time,value\n0,\n1,2\n", encoding="utf-8")
    result = kalman_track_csv(source, tmp_path / "run", time_column="time", observation_column="value",
                              observation_sd=1, process_noise=0, initial_rate_sd=0)
    validate(result, 18)
    metrics = pd.read_excel(result / "report.xlsx", sheet_name="Metrics")
    assert set(metrics["Target"]) == {"Observation"}
    assert metrics["Rows"].tolist() == [0, 0, 0]
    with pytest.raises(FileExistsError):
        kalman_track_csv(source, result, time_column="time", observation_column="value",
                         observation_sd=1, process_noise=0)


def test_pilot_stops_before_propagating_the_evaluation_rows():
    # Propagating over the huge interval would overflow the process covariance.
    trace = filter_series([0, 1, 2, 1e120], [1, 1, 8, 9], [0, 0, 0, 0],
                          observation_sd=1, process_noise=0.1, pilot_rows=2, min_rmse_gain=0.1)
    assert np.isnan(trace["forecast"][2:]).all()
    assert np.isnan(trace["filtered"][2:]).all()
    np.testing.assert_array_equal(trace["persistence"][1:], [1, 1, 8])


@pytest.mark.parametrize("future", ["3,3\n4,4\n5,5\n", "3,9\n4,-10\n5,11\n"])
def test_pilot_retains_kalman_without_using_later_observations(tmp_path, future):
    source = tmp_path / "series.csv"
    source.write_text("time,value\n0,0\n1,1\n2,2\n" + future, encoding="utf-8")
    settings = dict(time_column="time", observation_column="value", observation_sd=1, process_noise=0,
                    initial_rate=1, initial_rate_sd=0)
    result = kalman_track_csv(source, tmp_path / "pilot", **settings, pilot_rows=3, min_rmse_gain=0.1,
                              compact_report=True)
    full = kalman_track_csv(source, tmp_path / "full", **settings)
    validate(result, 18)
    book = pd.read_excel(result / "report.xlsx", sheet_name=None)
    assert set(book) == {"Metrics", "Summary", "Decision"}
    decision = book["Decision"].iloc[0]
    assert decision["Selected method"] == "Kalman"
    assert decision["Paired observations"] == 2
    assert decision["Relative RMSE gain"] == 1
    assert decision["Evaluation starts at"] == 3
    assert book["Metrics"]["Rows"].tolist() == [3, 3, 3]
    with np.load(result / "arrays.npz") as pilot, np.load(full / "arrays.npz") as all_rows:
        np.testing.assert_array_equal(pilot["forecast"], all_rows["forecast"])
        np.testing.assert_array_equal(pilot["filtered_covariance"], all_rows["filtered_covariance"])


@pytest.mark.parametrize(("pilot_readings", "reason"), [
    ("0,1\n1,1\n", "Last observation had zero error"),
    ("0,\n1,\n", "No paired pilot forecasts"),
])
def test_pilot_fallback_scores_only_the_available_later_baseline(tmp_path, pilot_readings, reason):
    source = tmp_path / "series.csv"
    source.write_text("time,value\n" + pilot_readings + "2,8\n3,10\n", encoding="utf-8")
    result = kalman_track_csv(source, tmp_path / "run", time_column="time", observation_column="value",
                              observation_sd=1, process_noise=0.1, pilot_rows=2, min_rmse_gain=0.1)
    validate(result, 18)
    book = pd.read_excel(result / "report.xlsx", sheet_name=None)
    assert book["Decision"].loc[0, "Selected method"] == "Last observation"
    assert book["Decision"].loc[0, "Reason"] == reason
    assert book["Metrics"]["Method"].tolist() == ["Last observation"]
    track = book["Track"]
    assert track["Forecast value"].iloc[2:].isna().all()
    assert track.loc[3, "Selected forecast"] == 8
    assert track.loc[3, "Forecast origin"] == 2
    assert track.loc[3, "Selected method"] == "Last observation"


def test_pilot_settings_require_both_parameters_and_an_evaluation_period():
    inputs = ([0, 1, 2], [0, 1, 2], [0, 0, 0])
    with pytest.raises(ValueError, match="together"):
        filter_series(*inputs, observation_sd=1, process_noise=0.1, pilot_rows=2)
    with pytest.raises(ValueError, match="evaluation row"):
        filter_series(*inputs, observation_sd=1, process_noise=0.1, pilot_rows=3, min_rmse_gain=0.1)
