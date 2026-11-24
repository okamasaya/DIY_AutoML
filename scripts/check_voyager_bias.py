"""Compare EKF bias states on paired Voyager-inspired synthetic observations."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag

from diy_automl import extended_kalman_filter
from make_voyager_demo import (
    AU_KM, BIAS_SIGMA, BIAS_START_DAY, OBSERVATION_STD, PRIOR_MEAN, PRIOR_STD, PROCESS_STD, TIME_DAYS,
    make_samples, measurement, measurement_jacobian, transition, transition_jacobian,
)


BIAS_MODELS = {"fixed_zero": (), "range": (0,), "range_rate": (1,), "both": (0, 1)}
BIAS_PRIOR_SD = 1.0  # In units of the channel's observation standard deviation.
BIAS_PROCESS_SD = 0.25  # In the same units, per seven-day step.
METRICS = ("position_rmse_km", "range_bias_rmse_sd", "range_rate_bias_rmse_sd")


def fit_orbit(observations, earth, bias_channels):
    """Propagate the orbit and a random walk for each selected bias channel."""
    channels = list(bias_channels)
    bias_projection = np.eye(len(OBSERVATION_STD))[:, channels]
    bias_noise = OBSERVATION_STD[channels]
    orbit_size = len(PRIOR_MEAN)
    bias_identity = np.eye(len(channels))
    trace = extended_kalman_filter(
        observations,
        initial_mean=np.r_[PRIOR_MEAN, np.zeros(len(channels))],
        initial_covariance=np.diag(np.r_[PRIOR_STD, BIAS_PRIOR_SD * bias_noise] ** 2),
        process_covariance=np.diag(np.r_[PROCESS_STD, BIAS_PROCESS_SD * bias_noise] ** 2),
        observation_covariance=np.diag(OBSERVATION_STD ** 2),
        transition=lambda state, step: np.r_[transition(state[:orbit_size], step), state[orbit_size:]],
        transition_jacobian=lambda state, step: block_diag(transition_jacobian(state[:orbit_size], step), bias_identity),
        measurement=lambda state, step: measurement(state[:orbit_size], earth[step - 1]) + bias_projection @ state[orbit_size:],
        measurement_jacobian=lambda state, step: np.column_stack((
            measurement_jacobian(state[:orbit_size], earth[step - 1]), bias_projection,
        )),
    )
    orbit = trace["filtered"][:, :orbit_size]
    bias_in_sd = (trace["filtered"][:, orbit_size:] @ bias_projection.T) / OBSERVATION_STD
    return orbit, bias_in_sd


def evaluate(seeds):
    after = TIME_DAYS[1:] >= BIAS_START_DAY
    periods = {"before": ~after, "after": after}
    scores = []
    squared_errors_by_group = defaultdict(list)
    for seed in seeds:
        truth, earth, observations = make_samples(seed)
        for scenario in ("nominal", "bias"):
            true_bias_in_sd = after[:, None] * BIAS_SIGMA if scenario == "bias" else np.zeros((len(after), 2))
            for model, channels in BIAS_MODELS.items():
                orbit, bias_in_sd = fit_orbit(observations[scenario], earth, channels)
                position_error = AU_KM * np.linalg.norm(orbit[:, :2] - truth[1:, :2], axis=1)
                errors = np.column_stack((position_error, bias_in_sd - true_bias_in_sd))
                for period, mask in periods.items():
                    mse = np.mean(errors[mask] ** 2, axis=0)
                    scores.append({
                        "seed": seed, "scenario": scenario, "model": model, "period": period,
                        **dict(zip(METRICS, np.sqrt(mse).tolist(), strict=True)),
                    })
                    squared_errors_by_group[scenario, model, period].append(mse)

    # All seeds contribute the same number of time steps to each group.
    pooled = [
        {"scenario": scenario, "model": model, "period": period,
         **dict(zip(METRICS, np.sqrt(np.mean(mse, axis=0)).tolist(), strict=True))}
        for (scenario, model, period), mse in squared_errors_by_group.items()
    ]
    return {
        "seeds": list(seeds),
        "bias_start_day": BIAS_START_DAY,
        "injected_bias_sd": BIAS_SIGMA.tolist(),
        "steps_by_period": {period: int(mask.sum()) for period, mask in periods.items()},
        "bias_prior_sd": BIAS_PRIOR_SD,
        "bias_process_sd_per_step": BIAS_PROCESS_SD,
        "per_seed": scores,
        "pooled": pooled,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New JSON result file")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
    args = parser.parse_args()
    if min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must contain distinct nonnegative integers")
    report = evaluate(args.seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
