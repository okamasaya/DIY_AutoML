import numpy as np
import pandas as pd
import pytest

from diy_automl import bootstrap_particle_filter
from diy_automl.ekf_benchmark import run_particle_benchmark, run_ukf_benchmark
from diy_automl.reporting import file_sha256, write_json
from diy_automl.validation import validate


def test_gaussian_limit_with_one_missing_measurement_component():
    result = bootstrap_particle_filter(
        [[np.nan, 1]], initial_mean=[0], initial_covariance=[[1]], transition=lambda x, k: x,
        measurement=lambda x, k: np.column_stack([2 * x[:, 0], x[:, 0]]), process_covariance=[[0.25]],
        observation_covariance=[[2, 0.3], [0.3, 0.75]], particles=16000,
    )
    # P^- = 1.25, K = 1.25/(1.25 + 0.75), conditional variance = (1-K)P^-.
    assert result["forecast_covariance"][0, 0, 0] == pytest.approx(1.25, abs=0.04)
    assert result["filtered"][0, 0] == pytest.approx(0.625, abs=0.025)
    assert result["filtered_covariance"][0, 0, 0] == pytest.approx(0.46875, abs=0.02)
    assert result["predicted_observation_covariance"][0, 1, 1] == pytest.approx(2, abs=0.04)


def test_square_measurement_retains_two_signs_before_resampling():
    result = bootstrap_particle_filter(
        [[4]], initial_mean=[0], initial_covariance=[[4]], transition=lambda x, k: x,
        measurement=lambda x, k: x**2, process_covariance=[[0]], observation_covariance=[[0.04]],
        particles=8000, record_steps=(1,),
    )
    points, weights = result["recorded_particles"][0, :, 0], result["recorded_weights"][0]
    assert weights.sum() == pytest.approx(1)
    assert weights[(np.abs(points) > 1.8) & (np.abs(points) < 2.2)].sum() > 0.99
    negative = weights[points < 0].sum()
    assert 0.35 < negative < 0.65
    assert result["filtered_negative_mass"][0, 0] == pytest.approx(negative)
    assert result["effective_sample_size"][0] == pytest.approx(1 / (weights @ weights))
    assert result["resampled"][0]
    assert result["filtered_lower"][0, 0] < -1.8 < 1.8 < result["filtered_upper"][0, 0]


def test_forecasts_precede_current_observations_and_snapshots_do_not_change_them():
    settings = dict(initial_mean=[0], initial_covariance=[[1]], transition=lambda x, k: x,
                    measurement=lambda x, k: x, process_covariance=[[0.1]], observation_covariance=[[0.5]])
    base = bootstrap_particle_filter([[np.nan], [1], [2]], **settings)
    recorded = bootstrap_particle_filter([[np.nan], [1], [2]], **settings, record_steps=(2,))
    altered = bootstrap_particle_filter([[np.nan], [-2], [3]], **settings)
    assert "recorded_particles" not in base
    for field, values in base.items():
        np.testing.assert_array_equal(values, recorded[field])
    for field in ("forecast", "forecast_covariance", "predicted_observation_lower", "predicted_observation_upper"):
        np.testing.assert_array_equal(base[field][:2], altered[field][:2])
    np.testing.assert_array_equal(base["filtered"][0], base["forecast"][0])
    assert not base["resampled"][0]
    assert base["filtered"][1, 0] != altered["filtered"][1, 0]


def test_particle_benchmark_preserves_comparison_inputs_and_replays_snapshots(tmp_path):
    settings = dict(seed=42, steps=8, trajectories=2)
    previous = run_ukf_benchmark(tmp_path / "ukf", **settings)
    result = run_particle_benchmark(tmp_path / "pf", **settings, particles=64)
    manifest = validate(result, 21)
    book = pd.read_excel(result / "report.xlsx", sheet_name=None)
    assert manifest["settings"] == {**settings, "particles": 64}
    assert book["Metrics"]["Rows"].tolist() == [16] * 18
    assert book["Particle_Diagnostics"]["Particles"].tolist() == [64, 64]
    artifact = result / "arrays.npz"
    with np.load(previous / "arrays.npz", allow_pickle=False) as old, np.load(artifact, allow_pickle=False) as current:
        for key in old.files:
            np.testing.assert_array_equal(current[key], old[key])
        np.testing.assert_array_equal(current["gordon__pf__recorded_steps"], [8])
        assert current["gordon__pf__recorded_particles"].shape == (1, 64, 1)
        arrays = dict(current)

    arrays["gordon__pf__recorded_weights"][0, 0] += 0.1
    np.savez_compressed(artifact, **arrays)
    manifest["artifacts"][artifact.name] = file_sha256(artifact)
    write_json(result / "manifest.json", manifest)
    with pytest.raises(ValueError, match="array differs: gordon__pf__recorded_weights"):
        validate(result, 21)
