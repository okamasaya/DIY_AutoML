"""Reproducible scalar benchmarks for first-order nonlinear state estimation."""

from dataclasses import dataclass
from numbers import Integral
from typing import Callable

import numpy as np
import pandas as pd

from .extended_kalman import extended_kalman_filter
from .kalman_workflow import METRIC_COLUMNS, interval, measure
from .reporting import fail, prepare, save, summary


SCHEMA = 19
ScalarFunction = Callable[[float | np.ndarray, int], float | np.ndarray]


@dataclass(frozen=True)
class ScalarModel:
    """A known scalar transition, measurement and noise model."""

    transition: ScalarFunction
    transition_derivative: ScalarFunction
    measurement: ScalarFunction
    measurement_derivative: ScalarFunction
    process_variance: float
    observation_variance: float
    initial_variance: float
    fixed_initial_state: float | None = None


CASES = {
    "monotone": ScalarModel(
        transition=lambda x, k: 0.85 * x + 0.2 * np.sin(x) + 0.1 * np.sin(0.2 * (k - 1)),
        transition_derivative=lambda x, k: 0.85 + 0.2 * np.cos(x),
        measurement=lambda x, k: x + 0.1 * x**3,
        measurement_derivative=lambda x, k: 1 + 0.3 * x**2,
        process_variance=0.04, observation_variance=0.04, initial_variance=0.25,
    ),
    # Gordon, Salmond and Smith (1993), section 4.1.
    "gordon": ScalarModel(
        transition=lambda x, k: 0.5 * x + 25 * x / (1 + x**2) + 8 * np.cos(1.2 * (k - 1)),
        transition_derivative=lambda x, k: 0.5 + 25 * (1 - x**2) / (1 + x**2)**2,
        measurement=lambda x, k: x**2 / 20,
        measurement_derivative=lambda x, k: x / 10,
        process_variance=10.0, observation_variance=1.0, initial_variance=2.0, fixed_initial_state=0.1,
    ),
}


def benchmark_settings(seed, steps, trajectories):
    for name, value, minimum in (("seed", seed, 0), ("steps", steps, 1), ("trajectories", trajectories, 1)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer at least {minimum}")
    return {"seed": int(seed), "steps": int(steps), "trajectories": int(trajectories)}


def simulate(model, rng, steps, trajectories):
    if model.fixed_initial_state is None:
        start = rng.normal(scale=np.sqrt(model.initial_variance), size=trajectories)
    else:
        start = np.full(trajectories, model.fixed_initial_state)
    truth = np.empty((trajectories, steps))
    observed = np.empty_like(truth)
    disturbances = rng.normal(scale=np.sqrt(model.process_variance), size=truth.shape)
    noise = rng.normal(scale=np.sqrt(model.observation_variance), size=truth.shape)
    state = start
    for index in range(steps):
        state = model.transition(state, index + 1) + disturbances[:, index]
        truth[:, index] = state
        observed[:, index] = model.measurement(state, index + 1) + noise[:, index]
    return {"initial_truth": start, "truth": truth, "observation": observed}


def estimate(model, observed):
    traces = [extended_kalman_filter(
        sequence[:, None], initial_mean=[0.0], initial_covariance=[[model.initial_variance]],
        transition=lambda x, k: [model.transition(x[0], k)],
        transition_jacobian=lambda x, k: [[model.transition_derivative(x[0], k)]],
        measurement=lambda x, k: [model.measurement(x[0], k)],
        measurement_jacobian=lambda x, k: [[model.measurement_derivative(x[0], k)]],
        process_covariance=[[model.process_variance]], observation_covariance=[[model.observation_variance]],
    ) for sequence in observed]
    return {name: np.stack([trace[name] for trace in traces]) for name in traces[0]}


def score_ekf(data):
    for stage, target, truth, prediction, covariance in (
        ("Forecast", "State", "truth", "forecast", "forecast_covariance"),
        ("Forecast", "Observation", "observation", "predicted_observation", "predicted_observation_covariance"),
        ("Filtered", "State", "truth", "filtered", "filtered_covariance"),
    ):
        values, variance = data[prediction][..., 0], data[covariance][..., 0, 0]
        yield measure(stage, target, "EKF", data[truth], values, np.isfinite(data[truth]), interval(values, variance))


def benchmark_results(settings):
    arrays, metrics, by_trajectory = {}, [], []
    seeds = np.random.SeedSequence(settings["seed"]).spawn(len(CASES))
    for (name, model), seed in zip(CASES.items(), seeds, strict=True):
        simulation = simulate(model, np.random.default_rng(seed), settings["steps"], settings["trajectories"])
        data = {**simulation, **estimate(model, simulation["observation"])}
        arrays.update({f"{name}__{key}": values for key, values in data.items()})
        metrics.extend([name, *row] for row in score_ekf(data))
        for index in range(settings["trajectories"]):
            trajectory = {key: values[index] for key, values in data.items()}
            by_trajectory.extend([name, index + 1, *row] for row in score_ekf(trajectory))
    return arrays, {
        "Metrics": pd.DataFrame(metrics, columns=["Case", *METRIC_COLUMNS]),
        "By_Trajectory": pd.DataFrame(by_trajectory, columns=["Case", "Trajectory", *METRIC_COLUMNS]),
        "Summary": summary([
            ("Seed", str(settings["seed"])), ("Steps per trajectory", settings["steps"]),
            ("Trajectories per case", settings["trajectories"]), ("Initial EKF mean", 0),
            *[(f"{name} Q, R, P0", f"{model.process_variance:g}, {model.observation_variance:g}, {model.initial_variance:g}")
              for name, model in CASES.items()],
            ("Forecast", "One step ahead using observations through the previous step"),
            ("Filtered", "State estimate after the current observation"),
            ("Intervals", "Pointwise 95% intervals from the EKF Gaussian approximation"),
        ]),
    }


def run_ekf_benchmark(output, *, seed=42, steps=50, trajectories=32):
    """Run both nonlinear benchmarks without fitting noise levels or selecting trajectories."""
    settings = benchmark_settings(seed, steps, trajectories)
    directory, manifest = prepare(output, SCHEMA, settings)
    try:
        arrays, sheets = benchmark_results(settings)
        return save(directory, manifest, sheets, arrays)
    except Exception as error:
        fail(directory, manifest, error)
        raise


def validate_ekf_benchmark(manifest, arrays, sheets):
    """Replay the simulation and filter from the recorded configuration."""
    from .ul_subsample_validation import same_table

    settings = benchmark_settings(**manifest["settings"])
    expected_arrays, expected_sheets = benchmark_results(settings)
    if set(expected_arrays) != set(arrays.files) or set(expected_sheets) != set(sheets):
        raise ValueError("EKF benchmark evidence differs")
    for name, expected in expected_arrays.items():
        actual = arrays[name]
        if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=1e-12, atol=1e-12):
            raise ValueError(f"EKF benchmark array differs: {name}")
    for name, expected in expected_sheets.items():
        same_table(sheets[name], expected, f"EKF benchmark sheet differs: {name}")
