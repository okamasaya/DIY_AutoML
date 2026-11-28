"""Time per-row EKF calls on a fixed four-state, two-observation model."""

import argparse
import json
from pathlib import Path
import platform
from time import perf_counter_ns

import numpy as np
from threadpoolctl import threadpool_limits

import uc_automail as uc_al


def measure(update, observations, warmup):
    durations = np.empty(len(observations) - warmup)
    for row in observations[:warmup]:
        update(row)
    for index, row in enumerate(observations[warmup:]):
        start = perf_counter_ns()
        update(row)
        durations[index] = (perf_counter_ns() - start) / 1000
    return {
        "median_us": float(np.median(durations)),
        "p95_us": float(np.percentile(durations, 95)),
        "p99_us": float(np.percentile(durations, 99)),
        "max_us": float(durations.max()),
        "calls_at_or_above_1ms": int(np.count_nonzero(durations >= 1000)),
    }


def benchmark():
    warmup, samples = 200, 6000
    observations = np.random.default_rng(42).normal(size=(warmup + samples, 2))
    transition = np.eye(4)
    transition[0, 2] = transition[1, 3] = 0.001
    measurement = np.eye(4)[:2]
    settings = dict(
        transition=lambda state, step: transition @ state, transition_jacobian=lambda state, step: transition,
        measurement=lambda state, step: measurement @ state, measurement_jacobian=lambda state, step: measurement,
        process_covariance=1e-3 * np.eye(4), observation_covariance=0.05 * np.eye(2),
    )
    initial_mean, initial_covariance = np.zeros(4), np.eye(4)
    mean, covariance = initial_mean, initial_covariance

    def one_row_batch(observation):
        nonlocal mean, covariance
        estimate = uc_al.extended_kalman_filter(
            observation[None, :], initial_mean=mean, initial_covariance=covariance, **settings,
        )
        mean, covariance = estimate["filtered"][0], estimate["filtered_covariance"][0]

    with threadpool_limits(limits=1, user_api="blas"):
        timings = {"one_row_batch": measure(one_row_batch, observations, warmup)}
        if hasattr(uc_al, "ExtendedKalmanFilter"):
            streaming = uc_al.ExtendedKalmanFilter(
                initial_mean=initial_mean, initial_covariance=initial_covariance, **settings,
            )
            timings["stateful_step"] = measure(streaming.step, observations, warmup)
            control_effect = 0.001 * np.array([0, 0, 1, 0])
            controlled_settings = {
                **settings,
                "transition": lambda state, step, u: transition @ state + control_effect * u,
                "transition_jacobian": lambda state, step, u: transition,
            }
            controlled = uc_al.ExtendedKalmanFilter(
                initial_mean=initial_mean, initial_covariance=initial_covariance, **controlled_settings,
            )
            timings["stateful_step_with_control"] = measure(
                lambda row: controlled.step(row, control=0.1), observations, warmup,
            )
    return {
        "platform": platform.system(), "python_version": platform.python_version(), "numpy_version": np.__version__,
        "package_version": uc_al.__version__, "blas_threads": 1, "warmup_calls": warmup,
        "timed_calls_per_method": samples, "states": 4, "observation_channels": 2, "timings": timings,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    contents = json.dumps(benchmark(), indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.write_text(contents, encoding="utf-8")
    else:
        print(contents, end="")


if __name__ == "__main__":
    main()
