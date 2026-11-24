"""Compare filters on observed one-step outcomes under a common local-level model."""

import numpy as np

from .extended_kalman import extended_kalman_filter
from .kalman_workflow import interval
from .particle_filter import bootstrap_particle_filter


def _observation_score(method, actual, predicted, bounds=None):
    result = {"method": method, "rmse": float(np.sqrt(np.mean((predicted - actual) ** 2)))}
    if bounds is not None:
        lower, upper = bounds
        result.update(
            coverage=float(np.mean((lower <= actual) & (actual <= upper))),
            mean_interval_width=float(np.mean(upper - lower)),
        )
    return result


def compare_local_level(values, calibration_rows, particles, *, particle_seeds=(42,), season=None):
    """Use fixed noise from the calibration prefix and score causal forecasts on later rows."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not 2 <= calibration_rows < len(values) or not np.isfinite(values).all():
        raise ValueError("Provide a finite scalar series and a calibration prefix of at least two rows")
    if season is not None and not 1 <= season < calibration_rows:
        raise ValueError("Seasonal lag must be shorter than the calibration prefix")

    change_variance = float(np.mean(np.diff(values[:calibration_rows]) ** 2))
    process_variance, observation_variance = 0.1 * change_variance, 0.45 * change_variance
    if not observation_variance > 0 or not np.isfinite(change_variance):
        raise ValueError("The calibration prefix must have finite nonzero changes")
    readings = values[1:, None]
    settings = dict(
        initial_mean=[values[0]],
        initial_covariance=[[observation_variance]],
        transition=lambda x, k: x,
        measurement=lambda x, k: x,
        process_covariance=[[process_variance]],
        observation_covariance=[[observation_variance]],
    )
    kalman = extended_kalman_filter(
        readings, **settings, transition_jacobian=lambda x, k: [[1]], measurement_jacobian=lambda x, k: [[1]],
    )
    evaluation = slice(calibration_rows - 1, None)
    actual = readings[evaluation, 0]
    forecast = kalman["predicted_observation"][evaluation, 0]
    bounds = interval(forecast, kalman["predicted_observation_covariance"][evaluation, 0, 0])
    metrics = [
        _observation_score("Last observation", actual, values[:-1][evaluation]),
        _observation_score("Linear Kalman", actual, forecast, bounds),
    ]
    if season is not None:
        seasonal = values[calibration_rows - season : -season]
        drift = np.mean(values[season:calibration_rows] - values[: calibration_rows - season])
        metrics.append(_observation_score("Seasonal naive", actual, seasonal))
        metrics.append(_observation_score("Seasonal naive with drift", actual, seasonal + drift))

    particle_runs = []
    for seed in particle_seeds:
        result = bootstrap_particle_filter(readings, **settings, particles=particles, seed=seed)
        forecast = result["predicted_observation"][evaluation, 0]
        bounds = (
            result["predicted_observation_lower"][evaluation, 0],
            result["predicted_observation_upper"][evaluation, 0],
        )
        particle_runs.append({"seed": seed, **_observation_score("Particle filter", actual, forecast, bounds)})
    return {
        "process_variance": process_variance,
        "observation_variance": observation_variance,
        "metrics": metrics,
        "particle_runs": particle_runs,
    }
