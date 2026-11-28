import numpy as np
import pytest

import uc_automail as uc_al


def test_nis_uses_the_available_covariance_and_preserves_missing_lags():
    observations = np.array([[2.0, 1.0], [np.nan, 2.0], [4.0, np.nan], [np.nan, np.nan]])
    result = {
        "predicted_observation": np.zeros_like(observations),
        "predicted_observation_covariance": np.array([[4.0, 1.0], [1.0, 1.0]]),
    }
    diagnostics = uc_al.filter_diagnostics(result, observations, max_lag=2)
    np.testing.assert_allclose(diagnostics["nis"], [4 / 3, 4, 4, np.nan])
    np.testing.assert_array_equal(diagnostics["observed_dimensions"], [2, 1, 1, 0])
    np.testing.assert_allclose(diagnostics["normalized_innovation"][0], [1.0, 1.0])
    np.testing.assert_allclose(diagnostics["nis_reference_interval"][:2, 1], [7.3777589, 5.0238862])
    np.testing.assert_array_equal(diagnostics["autocorrelation_pairs"], [[0, 1], [1, 0]])
    assert np.isnan(diagnostics["autocorrelation"][0, 0])
    assert diagnostics["autocorrelation"][1, 0] == pytest.approx(-0.5)
    assert np.isnan(diagnostics["ljung_box_pvalue"]).all()
    assert diagnostics["ljung_box_status"] == ("missing_observations", "missing_observations")
    assert np.isnan(diagnostics["autocorrelation_reference_limit"]).all()
    assert all(not lags.size for lags in diagnostics["autocorrelation_lags_outside_reference"])


def test_ljung_box_distinguishes_serial_correlation_from_innovation_magnitude():
    rng = np.random.default_rng(42)
    white = rng.normal(size=2048)
    correlated = white.copy()
    for index in range(1, len(correlated)):
        correlated[index] = 0.5 * correlated[index - 1] + np.sqrt(0.75) * white[index]
    result = {
        "predicted_observation": np.zeros((len(white), 1)),
        "predicted_observation_covariance": np.eye(1),
    }
    white_stats = uc_al.filter_diagnostics(result, white[:, None], max_lag=5)
    correlated_stats = uc_al.filter_diagnostics(result, correlated[:, None], max_lag=5)
    assert white_stats["mean_nis_per_dimension"] == pytest.approx(1, abs=0.1)
    assert correlated_stats["mean_nis_per_dimension"] == pytest.approx(1, abs=0.1)
    assert abs(white_stats["autocorrelation"][0, 0]) < 0.1
    assert correlated_stats["autocorrelation"][0, 0] > 0.4
    assert white_stats["ljung_box_pvalue"][0] > 0.05
    assert correlated_stats["ljung_box_pvalue"][0] < 0.001
    assert correlated_stats["ljung_box_status"] == ("ok",)
    assert correlated_stats["autocorrelation_reference_limit"][0] == pytest.approx(1.95996398454 / np.sqrt(len(white)))
    assert 1 in correlated_stats["autocorrelation_lags_outside_reference"][0]


def test_pull_and_nees_handle_partial_truth_and_singular_covariance():
    truth = np.array([[0.0, 0.0], [np.nan, 1.0], [0.0, 1.0]])
    result = {
        "filtered": np.array([[2.0, -1.0], [3.0, 2.0], [2.0, 1.0]]),
        "filtered_covariance": np.array([np.diag([4.0, 1.0]), np.diag([4.0, 1.0]), np.diag([0.0, 1.0])]),
    }
    diagnostics = uc_al.evaluate_filter_consistency(truth, result)
    np.testing.assert_allclose(diagnostics["pull"], [[1, -1], [np.nan, 1], [np.nan, 0]])
    np.testing.assert_allclose(diagnostics["pull_mean"], [1, 0])
    np.testing.assert_allclose(diagnostics["pull_std"], [np.nan, 1])
    np.testing.assert_allclose(diagnostics["pull_rms"], [1, np.sqrt(2 / 3)])
    np.testing.assert_allclose(diagnostics["nees"], [2, 1, np.nan])
    np.testing.assert_array_equal(diagnostics["nees_dimensions"], [2, 1, 0])


def test_single_missing_observation_has_no_scores_or_lags():
    result = {"predicted_observation": [[0.0]], "predicted_observation_covariance": [[1.0]]}
    diagnostics = uc_al.filter_diagnostics(result, [[np.nan]])
    assert np.isnan(diagnostics["nis"][0])
    assert np.isnan(diagnostics["mean_nis_per_dimension"])
    assert diagnostics["lags"].size == 0
    assert np.isnan(diagnostics["ljung_box_pvalue"][0])
    assert diagnostics["ljung_box_status"] == ("no_lags",)
    with pytest.raises(ValueError, match="confidence"):
        uc_al.filter_diagnostics(result, [[0.0]], confidence=1)


def test_nonzero_constant_innovations_have_no_autocorrelation_or_pvalue():
    observations = np.full((25, 1), 0.1)
    result = {
        "predicted_observation": np.zeros_like(observations),
        "predicted_observation_covariance": np.eye(1),
    }
    diagnostics = uc_al.filter_diagnostics(result, observations, max_lag=2)
    np.testing.assert_allclose(diagnostics["nis"], 0.01)
    np.testing.assert_array_equal(diagnostics["autocorrelation_pairs"][:, 0], [24, 23])
    assert np.isnan(diagnostics["autocorrelation"]).all()
    assert np.isnan(diagnostics["ljung_box_pvalue"]).all()
    assert diagnostics["ljung_box_status"] == ("constant_series",)
