"""Reproducible scalar benchmarks for nonlinear state estimation."""

from dataclasses import dataclass
from numbers import Integral
from time import perf_counter
from typing import Callable

import numpy as np
import pandas as pd

from .extended_kalman import extended_kalman_filter
from .kalman_workflow import METRIC_COLUMNS, interval, measure
from .reporting import fail, prepare, save, summary
from .unscented_kalman import unscented_kalman_filter


SCHEMA = 19
COMPARISON_SCHEMA = 20
FILTERS = {"EKF": extended_kalman_filter, "UKF": unscented_kalman_filter}
TIMING_COLUMNS = ["Case", "Method", "Filter seconds"]
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


def estimate(model, observed, method):
    settings = dict(initial_mean=[0.0], initial_covariance=[[model.initial_variance]],
                    transition=lambda x, k: [model.transition(x[0], k)],
                    measurement=lambda x, k: [model.measurement(x[0], k)],
                    process_covariance=[[model.process_variance]], observation_covariance=[[model.observation_variance]])
    if method == "EKF":
        settings.update(transition_jacobian=lambda x, k: [[model.transition_derivative(x[0], k)]],
                        measurement_jacobian=lambda x, k: [[model.measurement_derivative(x[0], k)]])
    traces = [FILTERS[method](sequence[:, None], **settings) for sequence in observed]
    return {name: np.stack([trace[name] for trace in traces]) for name in traces[0]}


def score_filter(data, method):
    for stage, target, truth, prediction, covariance in (
        ("Forecast", "State", "truth", "forecast", "forecast_covariance"),
        ("Forecast", "Observation", "observation", "predicted_observation", "predicted_observation_covariance"),
        ("Filtered", "State", "truth", "filtered", "filtered_covariance"),
    ):
        values, variance = data[prediction][..., 0], data[covariance][..., 0, 0]
        yield measure(stage, target, method, data[truth], values, np.isfinite(data[truth]), interval(values, variance))


def benchmark_results(settings, *, compare=False, time_filters=False):
    arrays, metrics, by_trajectory, timing = {}, [], [], []
    methods = FILTERS if compare else ("EKF",)
    seeds = np.random.SeedSequence(settings["seed"]).spawn(len(CASES))
    for (name, model), seed in zip(CASES.items(), seeds, strict=True):
        simulation = simulate(model, np.random.default_rng(seed), settings["steps"], settings["trajectories"])
        arrays.update({f"{name}__{key}": values for key, values in simulation.items()})
        for method in methods:
            start = perf_counter()
            trace = estimate(model, simulation["observation"], method)
            if time_filters:
                timing.append([name, method, perf_counter() - start])
            prefix = f"{name}__{method.lower()}__" if compare else f"{name}__"
            arrays.update({f"{prefix}{key}": values for key, values in trace.items()})
            data = {**simulation, **trace}
            metrics.extend([name, *row] for row in score_filter(data, method))
            for index in range(settings["trajectories"]):
                trajectory = {key: values[index] for key, values in data.items()}
                by_trajectory.extend([name, index + 1, *row] for row in score_filter(trajectory, method))
    sheets = {
        "Metrics": pd.DataFrame(metrics, columns=["Case", *METRIC_COLUMNS]),
        "By_Trajectory": pd.DataFrame(by_trajectory, columns=["Case", "Trajectory", *METRIC_COLUMNS]),
        "Summary": summary([
            ("Seed", str(settings["seed"])), ("Steps per trajectory", settings["steps"]),
            ("Trajectories per case", settings["trajectories"]), ("Initial filter mean" if compare else "Initial EKF mean", 0),
            *[(f"{name} Q, R, P0", f"{model.process_variance:g}, {model.observation_variance:g}, {model.initial_variance:g}")
              for name, model in CASES.items()],
            ("Forecast", "One step ahead using observations through the previous step"),
            ("Filtered", "State estimate after the current observation"),
            ("Intervals", f"Pointwise 95% intervals from the {'filter' if compare else 'EKF'} Gaussian approximation"),
            *([("UKF alpha, beta, kappa", "1, 2, 0")] if compare else []),
        ]),
    }
    if time_filters:
        sheets["Timing"] = pd.DataFrame(timing, columns=TIMING_COLUMNS)
    return arrays, sheets


def run_ekf_benchmark(output, *, seed=42, steps=50, trajectories=32):
    """Run both nonlinear benchmarks without fitting noise levels or selecting trajectories."""
    return run_benchmark(output, SCHEMA, benchmark_settings(seed, steps, trajectories))


def run_ukf_benchmark(output, *, seed=42, steps=50, trajectories=32):
    """Compare both filters with identical simulated states, observations and noise settings."""
    return run_benchmark(output, COMPARISON_SCHEMA, benchmark_settings(seed, steps, trajectories))


def run_benchmark(output, schema, settings):
    compare = schema == COMPARISON_SCHEMA
    directory, manifest = prepare(output, schema, settings)
    try:
        arrays, sheets = benchmark_results(settings, compare=compare, time_filters=compare)
        return save(directory, manifest, sheets, arrays)
    except Exception as error:
        fail(directory, manifest, error)
        raise


def validate_filter_benchmark(manifest, arrays, sheets):
    """Replay deterministic results and check recorded timing entries."""
    from .ul_subsample_validation import same_table

    settings = benchmark_settings(**manifest["settings"])
    compare = manifest["schema_version"] == COMPARISON_SCHEMA
    expected_arrays, expected_sheets = benchmark_results(settings, compare=compare)
    if compare:
        timing = sheets.get("Timing")
        pairs = [(case, method) for case in CASES for method in FILTERS]
        if timing is None or timing.columns.tolist() != TIMING_COLUMNS or len(timing) != len(pairs):
            raise ValueError("Filter timing table differs")
        seconds = timing["Filter seconds"].to_numpy(dtype=float)
        if (list(zip(timing["Case"], timing["Method"])) != pairs or not np.isfinite(seconds).all()
                or np.any(seconds < 0)):
            raise ValueError("Filter timing records differ")
        sheets = {name: table for name, table in sheets.items() if name != "Timing"}
    if set(expected_arrays) != set(arrays.files) or set(expected_sheets) != set(sheets):
        raise ValueError("EKF benchmark evidence differs")
    for name, expected in expected_arrays.items():
        actual = arrays[name]
        if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=1e-12, atol=1e-12):
            raise ValueError(f"EKF benchmark array differs: {name}")
    for name, expected in expected_sheets.items():
        same_table(sheets[name], expected, f"EKF benchmark sheet differs: {name}")
