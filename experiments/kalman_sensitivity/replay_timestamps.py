"""Replay missing and delayed position observations through the released EKF API."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import chi2

import uc_automail as uc_al


SEED = 20261128
TRAJECTORIES = 12
STEPS = 640
BURN_IN = 80
DT = 0.02
DELAY_STEPS = 8
PARTIAL_GAP = slice(145, 185)
FULL_GAP = slice(195, 225)
DELAY_WINDOW = slice(260, 340)

IDENTITY = np.eye(2)
ZERO = np.zeros((2, 2))
TRANSITION = np.block([[IDENTITY, DT * IDENTITY], [ZERO, IDENTITY]])
CONTROL = np.vstack((DT**2 / 2 * IDENTITY, DT * IDENTITY))
MEASUREMENT = np.hstack((IDENTITY, ZERO))
INITIAL_COVARIANCE = np.diag([0.04, 0.04, 0.04, 0.04])
PROCESS_COVARIANCE = 0.12 * np.kron(
    [[DT**3 / 3, DT**2 / 2], [DT**2 / 2, DT]], IDENTITY,
)
OBSERVATION_COVARIANCE = np.array([[0.06**2, 0.0012], [0.0012, 0.09**2]])


def generate_data():
    rng = np.random.default_rng(SEED)
    time = DT * np.arange(STEPS)
    controls = np.column_stack((np.sin(2 * time), 0.7 * np.cos(1.5 * time)))
    previous = rng.multivariate_normal(np.zeros(4), INITIAL_COVARIANCE, TRAJECTORIES)
    disturbances = rng.multivariate_normal(np.zeros(4), PROCESS_COVARIANCE, (STEPS, TRAJECTORIES))
    truth = np.empty((TRAJECTORIES, STEPS, 4))
    for index, control in enumerate(controls):
        previous = previous @ TRANSITION.T + CONTROL @ control + disturbances[index]
        truth[:, index] = previous
    noise = rng.multivariate_normal(np.zeros(2), OBSERVATION_COVARIANCE, (TRAJECTORIES, STEPS))
    return controls, truth, truth @ MEASUREMENT.T + noise


def align_packets(observations, *, use_stale):
    readings = observations.copy()
    readings[PARTIAL_GAP, 1] = np.nan
    readings[FULL_GAP] = np.nan
    arrival = np.arange(STEPS)
    arrival[DELAY_WINDOW] += DELAY_STEPS
    packets = [[] for _ in range(STEPS)]
    for timestamp, arrival_tick in enumerate(arrival):
        if not np.isnan(readings[timestamp]).all():
            packets[arrival_tick].append(timestamp)

    aligned = np.full_like(readings, np.nan)
    stale_used = 0
    for tick, timestamps in enumerate(packets):
        if timestamps:
            newest = timestamps[-1]
            if newest == tick or use_stale:
                aligned[tick] = readings[newest]
                stale_used += newest != tick
    return aligned, stale_used


def filter_on_grid(observations, controls, *, verify_batch):
    settings = dict(
        initial_mean=np.zeros(4), initial_covariance=INITIAL_COVARIANCE,
        transition=lambda x, k, u: TRANSITION @ x + CONTROL @ u,
        transition_jacobian=lambda x, k, u: TRANSITION,
        measurement=lambda x, k: MEASUREMENT @ x,
        measurement_jacobian=lambda x, k: MEASUREMENT,
        process_covariance=PROCESS_COVARIANCE, observation_covariance=OBSERVATION_COVARIANCE,
    )
    tracker = uc_al.ExtendedKalmanFilter(**settings)
    rows = [tracker.step(row, control=u) for row, u in zip(observations, controls, strict=True)]
    result = {name: np.stack([row[name] for row in rows]) for name in rows[0]}
    if verify_batch:
        batch = uc_al.extended_kalman_filter(observations, controls=controls, **settings)
        for name, values in batch.items():
            np.testing.assert_allclose(result[name], values, rtol=1e-12, atol=1e-12)

    missing = np.isnan(observations).all(axis=1)
    np.testing.assert_array_equal(result["filtered"][missing], result["forecast"][missing])
    np.testing.assert_array_equal(result["filtered_covariance"][missing], result["forecast_covariance"][missing])
    return result


def evaluate(controls, truth, readings, condition):
    errors, covered = [], []
    stale_count = partial_count = missing_count = 0
    statuses = set()
    ellipse_limit = chi2.ppf(0.95, 2)
    for reference, observations in zip(truth, readings, strict=True):
        if condition != "complete":
            observations, stale = align_packets(observations, use_stale=condition == "use_stale")
            stale_count += stale

        result = filter_on_grid(observations, controls, verify_batch=condition == "keep_timestamps")
        diagnostics = uc_al.filter_diagnostics(result, observations, max_lag=3)
        available = np.sum(np.isfinite(observations), axis=1)
        np.testing.assert_array_equal(diagnostics["observed_dimensions"], available)
        np.testing.assert_array_equal(np.isnan(diagnostics["nis"]), available == 0)
        statuses.update(diagnostics["ljung_box_status"])
        partial_count += int(np.count_nonzero(available == 1))
        missing_count += int(np.count_nonzero(available == 0))

        position_error = result["filtered"][:, :2] - reference[:, :2]
        covariance = result["filtered_covariance"][:, :2, :2]
        whitened = np.linalg.solve(np.linalg.cholesky(covariance), position_error[..., None])[..., 0]
        errors.append(position_error)
        covered.append(np.sum(whitened**2, axis=1) <= ellipse_limit)

    squared = np.asarray(errors)**2
    ellipses = np.asarray(covered)
    summary = {
        "partial_observations": partial_count,
        "prediction_only_steps": missing_count,
        "stale_packets_used": stale_count,
        "ljung_box_statuses": sorted(statuses),
    }
    for name, window in (("scored_period", slice(BURN_IN, None)), ("delayed_period", DELAY_WINDOW)):
        scores = squared[:, window]
        summary[name] = {
            "position_rmse_m": float(np.sqrt(scores.sum(axis=2).mean())),
            "axis_rmse_m": np.sqrt(scores.mean(axis=(0, 1))).tolist(),
            "position_ellipse_coverage": float(ellipses[:, window].mean()),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = generate_data()
    result = {
        "version": uc_al.__version__, "seed": SEED, "trajectories": TRAJECTORIES, "steps": STEPS,
        "burn_in_steps": BURN_IN, "step_seconds": DT, "delay_steps": DELAY_STEPS,
        "conditions": {name: evaluate(*data, condition=name)
                       for name in ("complete", "keep_timestamps", "use_stale")},
    }
    contents = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(contents, end="")
    else:
        args.output.write_text(contents, encoding="utf-8")


if __name__ == "__main__":
    main()
