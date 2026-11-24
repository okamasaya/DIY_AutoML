import numpy as np
import pandas as pd
import pytest

from diy_automl import unscented_kalman_filter
from diy_automl.ekf_benchmark import run_ekf_benchmark, run_ukf_benchmark
from diy_automl.reporting import file_sha256, write_json
from diy_automl.validation import validate


def test_quadratic_observation_includes_process_noise_with_exact_gaussian_moments():
    result = unscented_kalman_filter(
        [[4]], initial_mean=[0], initial_covariance=[[1]], transition=lambda x, k: x,
        measurement=lambda x, k: x**2, process_covariance=[[2]], observation_covariance=[[1]],
    )
    np.testing.assert_allclose(result["forecast_covariance"], [[[3]]])
    # For X ~ N(0, 3), E[X²]=3, Var(X²)=18, and Cov(X, X²)=0.
    np.testing.assert_allclose(result["predicted_observation"], [[3]])
    np.testing.assert_allclose(result["predicted_observation_covariance"], [[[19]]])
    np.testing.assert_allclose(result["filtered"], [[0]], atol=1e-14)
    np.testing.assert_allclose(result["filtered_covariance"], [[[3]]])


def test_rank_deficient_state_covariance_does_not_inject_noise():
    result = unscented_kalman_filter(
        [[3]], initial_mean=[1, 2], initial_covariance=[[0, 0], [0, 1]], transition=lambda x, k: x,
        measurement=lambda x, k: x[1:], process_covariance=np.zeros((2, 2)), observation_covariance=[[1]],
    )
    np.testing.assert_allclose(result["filtered"], [[1, 2.5]])
    np.testing.assert_allclose(result["filtered_covariance"], [[[0, 0], [0, 0.5]]], atol=1e-14)


def test_paired_benchmark_reuses_the_ekf_inputs_and_validates_ukf_outputs(tmp_path):
    options = dict(seed=42, steps=9, trajectories=2)
    earlier = run_ekf_benchmark(tmp_path / "ekf", **options)
    paired = run_ukf_benchmark(tmp_path / "paired", **options)
    manifest = validate(paired, 20)
    assert manifest["settings"] == options
    book = pd.read_excel(paired / "report.xlsx", sheet_name=None)
    assert book["Metrics"]["Rows"].tolist() == [18] * 12
    assert book["Timing"]["Method"].tolist() == ["EKF", "UKF", "EKF", "UKF"]
    artifact = paired / "arrays.npz"
    with np.load(earlier / "arrays.npz", allow_pickle=False) as old, np.load(artifact, allow_pickle=False) as data:
        for key in old.files:
            case, field = key.split("__", 1)
            paired_key = key if field in {"truth", "initial_truth", "observation"} else f"{case}__ekf__{field}"
            np.testing.assert_array_equal(old[key], data[paired_key])
        arrays = dict(data)

    arrays["gordon__ukf__filtered"][1, 0, 0] += 1
    np.savez_compressed(artifact, **arrays)
    manifest["artifacts"][artifact.name] = file_sha256(artifact)
    write_json(paired / "manifest.json", manifest)
    with pytest.raises(ValueError, match="array differs: gordon__ukf__filtered"):
        validate(paired, 20)
