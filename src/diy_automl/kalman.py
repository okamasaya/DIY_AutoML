"""Causal Kalman filtering for a scalar level and its rate of change."""

import numpy as np


def filter_series(time, observation, acceleration, *, observation_sd, process_noise,
                  initial_rate=0.0, initial_rate_sd=1.0):
    """Predict each scheduled time before updating with its observation.

    Acceleration at row i applies from time[i] to time[i + 1]. Missing observations
    are NaN. The process noise is the spectral density of white acceleration noise.
    """
    time, observation, acceleration = (np.asarray(values, dtype=float) for values in (time, observation, acceleration))
    if time.ndim != 1 or len(time) < 2 or observation.shape != time.shape or acceleration.shape != time.shape:
        raise ValueError("Provide at least two times and one observation and acceleration slot per time")
    if not np.isfinite(time).all() or not np.all(time[1:] > time[:-1]):
        raise ValueError("Times must be finite and strictly increasing")
    if np.isinf(observation).any() or not np.isfinite(acceleration).all():
        raise ValueError("Observations must be finite or missing, and accelerations must be finite")
    observed = np.flatnonzero(np.isfinite(observation))
    if not observed.size:
        raise ValueError("At least one observation is required to initialize the filter")
    observation_variance = observation_sd * observation_sd
    rate_variance = initial_rate_sd * initial_rate_sd
    if (observation_sd <= 0 or observation_variance <= 0 or initial_rate_sd < 0 or process_noise < 0
            or not np.isfinite([observation_variance, rate_variance, initial_rate, process_noise]).all()):
        raise ValueError("Use a positive observation SD, nonnegative process noise and initial rate SD, and a finite initial rate")

    first = observed[0]
    forecast = np.full((len(time), 2), np.nan)
    filtered = np.full_like(forecast, np.nan)
    dynamics = np.full_like(forecast, np.nan)
    forecast_covariance = np.full((len(time), 2, 2), np.nan)
    filtered_covariance = np.full_like(forecast_covariance, np.nan)
    persistence = np.full(len(time), np.nan)
    filtered[first] = dynamics[first] = (observation[first], initial_rate)
    filtered_covariance[first] = np.diag([observation_variance, rate_variance])
    last_observed = observation[first]
    identity = np.eye(2)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for i in range(first + 1, len(time)):
            step = time[i] - time[i - 1]
            transition = np.array([[1.0, step], [0.0, 1.0]])
            control = acceleration[i - 1] * np.array([0.5 * step * step, step])
            noise = process_noise * step * np.array([[step * step / 3, step / 2], [step / 2, 1.0]])
            predicted = transition @ filtered[i - 1] + control
            covariance = transition @ filtered_covariance[i - 1] @ transition.T + noise
            forecast[i] = predicted
            forecast_covariance[i] = covariance
            dynamics[i] = transition @ dynamics[i - 1] + control
            persistence[i] = last_observed
            if np.isnan(observation[i]):
                filtered[i], filtered_covariance[i] = predicted, covariance
                continue

            gain = covariance[:, 0] / (covariance[0, 0] + observation_variance)
            filtered[i] = predicted + gain * (observation[i] - predicted[0])
            correction = identity.copy()
            correction[:, 0] -= gain
            # The Joseph form preserves symmetry and positive semidefiniteness in exact arithmetic.
            updated = correction @ covariance @ correction.T + observation_variance * np.outer(gain, gain)
            filtered_covariance[i] = 0.5 * (updated + updated.T)
            last_observed = observation[i]

    return {"forecast": forecast, "filtered": filtered, "dynamics": dynamics, "persistence": persistence,
            "forecast_covariance": forecast_covariance, "filtered_covariance": filtered_covariance}
