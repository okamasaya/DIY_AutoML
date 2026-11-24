import numpy as np
import pandas as pd
import pytest

from diy_automl import extended_kalman_filter
from diy_automl.ekf_benchmark import CASES, run_ekf_benchmark
from diy_automl.kalman import filter_series
from diy_automl.reporting import file_sha256, write_json
from diy_automl.validation import validate


def test_ekf_matches_existing_linear_filter_with_irregular_steps_and_missing_observation():
    time = np.array([0, 0.5, 2.0, 4.0, 4.25])
    readings = np.array([0, 0.7, np.nan, 5.5, 6.0])
    commands = np.array([1.0, 0.2, 0.4, -0.2, 0.0])
    dt = np.diff(time)
    transitions = np.array([[[1, step], [0, 1]] for step in dt])
    process = np.array([0.3 * np.array([[step**3 / 3, step**2 / 2], [step**2 / 2, step]]) for step in dt])
    controls = commands[:-1, None] * np.column_stack([dt**2 / 2, dt])
    linear = filter_series(time, readings, commands, observation_sd=0.5, process_noise=0.3,
                           initial_rate=1, initial_rate_sd=2)
    ekf = extended_kalman_filter(
        readings[1:, None], initial_mean=[0, 1], initial_covariance=[[0.25, 0], [0, 4]],
        transition=lambda x, k: transitions[k - 1] @ x + controls[k - 1],
        transition_jacobian=lambda x, k: transitions[k - 1],
        measurement=lambda x, k: [x[0]], measurement_jacobian=lambda x, k: [[1, 0]],
        process_covariance=process, observation_covariance=[[0.25]],
    )
    for key in ("forecast", "filtered", "forecast_covariance", "filtered_covariance"):
        np.testing.assert_allclose(ekf[key], linear[key][1:], atol=1e-12, rtol=0)
    np.testing.assert_allclose(ekf["predicted_observation_covariance"][:, 0, 0],
                               ekf["forecast_covariance"][:, 0, 0] + 0.25)


def test_multivariate_update_uses_only_available_observation_components():
    matrix = np.array([[2, 1], [1, -1]])
    prior = np.array([[2, 0.5], [0.5, 1]])
    noise = np.array([[2, 0.3], [0.3, 3]])
    result = extended_kalman_filter(
        [[np.nan, 8], [np.nan, np.nan]], initial_mean=[0, 0], initial_covariance=prior,
        transition=lambda x, k: x, transition_jacobian=lambda x, k: np.eye(2),
        measurement=lambda x, k: matrix @ x, measurement_jacobian=lambda x, k: matrix,
        process_covariance=np.zeros((2, 2)), observation_covariance=noise,
    )
    np.testing.assert_array_equal(result["forecast"][0], [0, 0])
    np.testing.assert_allclose(result["predicted_observation_covariance"][0], matrix @ prior @ matrix.T + noise)
    np.testing.assert_allclose(result["filtered"][0], [2.4, -0.8])
    np.testing.assert_allclose(result["filtered_covariance"][0], [[1.55, 0.65], [0.65, 0.95]])
    np.testing.assert_allclose(result["filtered"][1], result["forecast"][1])
    np.testing.assert_allclose(result["filtered_covariance"][1], result["forecast_covariance"][1])


def test_squared_measurement_at_zero_has_zero_first_order_gain():
    result = extended_kalman_filter(
        [[4]], initial_mean=[0], initial_covariance=[[1]],
        transition=lambda x, k: x, transition_jacobian=lambda x, k: [[1]],
        measurement=lambda x, k: x**2, measurement_jacobian=lambda x, k: [[2 * x[0]]],
        process_covariance=[[0]], observation_covariance=[[1]],
    )
    np.testing.assert_array_equal(result["filtered"], result["forecast"])
    np.testing.assert_array_equal(result["filtered_covariance"], result["forecast_covariance"])


@pytest.mark.parametrize(("override", "message"), [
    ({"measurement_jacobian": lambda x, k: [1]}, "Measurement Jacobian must be finite"),
    ({"process_covariance": [[-1]]}, "Process covariance must be positive semidefinite"),
    ({"observation_covariance": [[0]]}, "Observation covariance must be positive definite"),
])
def test_invalid_nonlinear_model_inputs(override, message):
    settings = dict(initial_mean=[0], initial_covariance=[[1]], transition=lambda x, k: x,
                    transition_jacobian=lambda x, k: [[1]], measurement=lambda x, k: x,
                    measurement_jacobian=lambda x, k: [[1]], process_covariance=[[1]], observation_covariance=[[1]])
    with pytest.raises(ValueError, match=message):
        extended_kalman_filter([[1]], **{**settings, **override})


@pytest.mark.parametrize("model", CASES.values(), ids=CASES.keys())
def test_benchmark_jacobians_match_finite_differences(model):
    x = np.array([-7, -1.2, -0.01, 0, 1.3, 4.0])
    step, eps = 4, 1e-6
    for transform, derivative in ((model.transition, model.transition_derivative),
                                  (model.measurement, model.measurement_derivative)):
        finite_difference = (transform(x + eps, step) - transform(x - eps, step)) / (2 * eps)
        np.testing.assert_allclose(derivative(x, step), finite_difference, atol=2e-8, rtol=0)


def test_ekf_benchmark_report_and_replay(tmp_path):
    result = run_ekf_benchmark(tmp_path / "run", steps=12, trajectories=3)
    manifest = validate(result, 19)
    assert manifest["settings"] == {"seed": 42, "steps": 12, "trajectories": 3}
    report = pd.read_excel(result / "report.xlsx", sheet_name=None)
    assert report["Metrics"]["Rows"].tolist() == [36] * 6
    assert report["By_Trajectory"]["Rows"].tolist() == [12] * 18
    metric = report["Metrics"].set_index(["Case", "Stage", "Target"]).loc[("gordon", "Filtered", "State")]
    artifact = result / "arrays.npz"
    with np.load(artifact, allow_pickle=False) as stored:
        arrays = dict(stored)
    error = arrays["gordon__truth"] - arrays["gordon__filtered"][..., 0]
    margin = 1.959963984540054 * np.sqrt(arrays["gordon__filtered_covariance"][..., 0, 0])
    assert metric["RMSE"] == pytest.approx(np.sqrt(np.mean(error**2)))
    assert metric["Coverage"] == pytest.approx(np.mean(np.abs(error) <= margin))

    arrays["gordon__observation"][1, 2] += 1
    np.savez_compressed(artifact, **arrays)
    manifest["artifacts"][artifact.name] = file_sha256(artifact)
    write_json(result / "manifest.json", manifest)
    with pytest.raises(ValueError, match="EKF benchmark array differs: gordon__observation"):
        validate(result, 19)
