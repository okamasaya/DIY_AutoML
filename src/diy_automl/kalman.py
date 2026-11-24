"""Causal Kalman filtering for a scalar level and its rate of change."""

from numbers import Integral

import numpy as np


def assess_pilot(observation, forecast, persistence, min_rmse_gain):
    """Compare both forecasts on the same observed rows in the pilot prefix."""
    paired = np.isfinite(observation) & np.isfinite(forecast) & np.isfinite(persistence)
    result = {"Paired observations": int(paired.sum()), "Kalman RMSE": np.nan, "Last observation RMSE": np.nan,
              "Relative RMSE gain": np.nan, "Minimum gain": min_rmse_gain,
              "Selected method": "Last observation", "Reason": "No paired pilot forecasts"}
    if not result["Paired observations"]:
        return result
    kalman_rmse = np.sqrt(np.mean((forecast[paired] - observation[paired])**2))
    baseline_rmse = np.sqrt(np.mean((persistence[paired] - observation[paired])**2))
    result.update({"Kalman RMSE": kalman_rmse, "Last observation RMSE": baseline_rmse})
    if baseline_rmse == 0:
        result["Reason"] = "Last observation had zero error"
    else:
        gain = (baseline_rmse - kalman_rmse) / baseline_rmse
        use_kalman = gain > 0 and gain >= min_rmse_gain
        result.update({"Relative RMSE gain": gain, "Selected method": "Kalman" if use_kalman else "Last observation",
                       "Reason": "Minimum gain met" if use_kalman else "Minimum gain not met"})
    return result


def filter_series(time, observation, acceleration, *, observation_sd, process_noise,
                  initial_rate=0.0, initial_rate_sd=1.0, pilot_rows=None, min_rmse_gain=None):
    """Predict each scheduled time before updating with its observation.

    Acceleration at row i applies from time[i] to time[i + 1]. Missing observations
    are NaN. The process noise is the spectral density of white acceleration noise.
    With a pilot, the filter continues only if its relative RMSE gain meets the threshold.
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
    if (pilot_rows is None) != (min_rmse_gain is None):
        raise ValueError("Specify pilot rows and minimum RMSE gain together")
    if pilot_rows is not None and (isinstance(pilot_rows, bool) or not isinstance(pilot_rows, Integral)
                                   or not 2 <= pilot_rows < len(time) or not 0 <= min_rmse_gain <= 1):
        raise ValueError("Pilot rows must leave at least one evaluation row, with a minimum gain from 0 to 1")

    first = observed[0]
    forecast = np.full((len(time), 2), np.nan)
    filtered = np.full_like(forecast, np.nan)
    dynamics = np.full_like(forecast, np.nan)
    forecast_covariance = np.full((len(time), 2, 2), np.nan)
    filtered_covariance = np.full_like(forecast_covariance, np.nan)
    last = np.maximum.accumulate(np.where(np.isfinite(observation), np.arange(len(time)), -1))
    persistence = np.full(len(time), np.nan)
    persistence[1:] = np.where(last[:-1] >= 0, observation[last[:-1]], np.nan)
    trace = {"forecast": forecast, "filtered": filtered, "dynamics": dynamics, "persistence": persistence,
             "forecast_covariance": forecast_covariance, "filtered_covariance": filtered_covariance}
    if pilot_rows is not None and first >= pilot_rows:
        return trace
    filtered[first] = dynamics[first] = (observation[first], initial_rate)
    filtered_covariance[first] = np.diag([observation_variance, rate_variance])
    identity = np.eye(2)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for i in range(first + 1, len(time)):
            if i == pilot_rows:
                decision = assess_pilot(observation[:i], forecast[:i, 0], persistence[:i], min_rmse_gain)
                if decision["Selected method"] == "Last observation":
                    break
            step = time[i] - time[i - 1]
            transition = np.array([[1.0, step], [0.0, 1.0]])
            control = acceleration[i - 1] * np.array([0.5 * step * step, step])
            noise = process_noise * step * np.array([[step * step / 3, step / 2], [step / 2, 1.0]])
            predicted = transition @ filtered[i - 1] + control
            covariance = transition @ filtered_covariance[i - 1] @ transition.T + noise
            forecast[i] = predicted
            forecast_covariance[i] = covariance
            dynamics[i] = transition @ dynamics[i - 1] + control
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

    return trace
