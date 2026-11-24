"""Reproducible scalar benchmarks for nonlinear state estimation."""

from dataclasses import dataclass
from numbers import Integral
from time import perf_counter
from typing import Callable

import numpy as np
import pandas as pd

from .extended_kalman import extended_kalman_filter
from .kalman_workflow import METRIC_COLUMNS, interval, measure
from .particle_filter import bootstrap_particle_filter
from .reporting import fail, prepare, save, summary
from .unscented_kalman import unscented_kalman_filter


SCHEMA = 19
COMPARISON_SCHEMA = 20
PARTICLE_SCHEMA = 21
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


def benchmark_settings(seed, steps, trajectories, *, particles=None):
    for name, value, minimum in (("seed", seed, 0), ("steps", steps, 1), ("trajectories", trajectories, 1)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer at least {minimum}")
    settings = {"seed": int(seed), "steps": int(steps), "trajectories": int(trajectories)}
    if particles is not None:
        if isinstance(particles, bool) or not isinstance(particles, Integral) or particles < 2:
            raise ValueError("particles must be an integer at least 2")
        settings["particles"] = int(particles)
    return settings


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


def estimate_particles(model, observed, settings, case_index):
    seeds = np.random.SeedSequence([settings["seed"], case_index]).spawn(len(observed))
    traces = []
    snapshot = {}
    for index, (sequence, seed) in enumerate(zip(observed, seeds, strict=True)):
        trace = bootstrap_particle_filter(
            sequence[:, None], initial_mean=[0.0], initial_covariance=[[model.initial_variance]],
            transition=lambda x, k: model.transition(x, k), measurement=lambda x, k: model.measurement(x, k),
            process_covariance=[[model.process_variance]], observation_covariance=[[model.observation_variance]],
            particles=settings["particles"], seed=int(seed.generate_state(1, dtype=np.uint64)[0]),
            record_steps=(min(21, len(sequence)),) if index == 0 else (),
        )
        if index == 0:
            snapshot = {key: trace.pop(key) for key in ("recorded_steps", "recorded_particles", "recorded_weights")}
        traces.append(trace)
    return {name: np.stack([trace[name] for trace in traces]) for name in traces[0]}, snapshot


def score_filter(data, method):
    for stage, target, truth, prediction, covariance in (
        ("Forecast", "State", "truth", "forecast", "forecast_covariance"),
        ("Forecast", "Observation", "observation", "predicted_observation", "predicted_observation_covariance"),
        ("Filtered", "State", "truth", "filtered", "filtered_covariance"),
    ):
        values = data[prediction][..., 0]
        if method == "PF":
            bounds = (data[f"{prediction}_lower"][..., 0], data[f"{prediction}_upper"][..., 0])
        else:
            bounds = interval(values, data[covariance][..., 0, 0])
        yield measure(stage, target, method, data[truth], values, np.isfinite(data[truth]), bounds)


def benchmark_methods(settings, compare):
    if "particles" in settings:
        return (*FILTERS, "PF")
    return tuple(FILTERS) if compare else ("EKF",)


def benchmark_results(settings, *, compare=False, time_filters=False):
    arrays, metrics, by_trajectory, timing, diagnostics = {}, [], [], [], []
    methods = benchmark_methods(settings, compare)
    seeds = np.random.SeedSequence(settings["seed"]).spawn(len(CASES))
    for case_index, ((name, model), seed) in enumerate(zip(CASES.items(), seeds, strict=True)):
        simulation = simulate(model, np.random.default_rng(seed), settings["steps"], settings["trajectories"])
        arrays.update({f"{name}__{key}": values for key, values in simulation.items()})
        for method in methods:
            start = perf_counter()
            if method == "PF":
                trace, snapshot = estimate_particles(model, simulation["observation"], settings, case_index)
                arrays.update({f"{name}__pf__{key}": value for key, value in snapshot.items()})
            else:
                trace = estimate(model, simulation["observation"], method)
            if time_filters:
                timing.append([name, method, perf_counter() - start])
            if method == "PF":
                ess = trace["effective_sample_size"]
                negative = trace["filtered_negative_mass"][..., 0]
                diagnostics.append([name, settings["particles"], ess.mean(), ess.min(), int(trace["resampled"].sum()),
                                    int(np.count_nonzero((negative >= 0.1) & (negative <= 0.9)))])
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
            ("Intervals", "Pointwise 95% Gaussian intervals for EKF and UKF" if diagnostics else
             f"Pointwise 95% intervals from the {'filter' if compare else 'EKF'} Gaussian approximation"),
            *([("UKF alpha, beta, kappa", "1, 2, 0")] if compare else []),
            *([("PF intervals", "Weighted 2.5th and 97.5th percentiles"), ("PF resampling", "Systematic at ESS below N/2"),
               ("PF snapshot", "First trajectory, step 21 or final step if shorter")] if diagnostics else []),
        ]),
    }
    if time_filters:
        sheets["Timing"] = pd.DataFrame(timing, columns=TIMING_COLUMNS)
    if diagnostics:
        sheets["Particle_Diagnostics"] = pd.DataFrame(diagnostics, columns=[
            "Case", "Particles", "Mean ESS", "Minimum ESS", "Resampled steps", "Steps with 0.1 <= negative mass <= 0.9",
        ])
    return arrays, sheets


def run_ekf_benchmark(output, *, seed=42, steps=50, trajectories=32):
    """Run both nonlinear benchmarks without fitting noise levels or selecting trajectories."""
    return run_benchmark(output, SCHEMA, benchmark_settings(seed, steps, trajectories))


def run_ukf_benchmark(output, *, seed=42, steps=50, trajectories=32):
    """Compare both filters with identical simulated states, observations and noise settings."""
    return run_benchmark(output, COMPARISON_SCHEMA, benchmark_settings(seed, steps, trajectories))


def run_particle_benchmark(output, *, seed=42, steps=50, trajectories=32, particles=256):
    """Compare EKF, UKF and the bootstrap particle filter on identical sequences."""
    if particles is None:
        raise ValueError("particles must be an integer at least 2")
    return run_benchmark(output, PARTICLE_SCHEMA, benchmark_settings(seed, steps, trajectories, particles=particles))


def run_benchmark(output, schema, settings):
    compare = schema != SCHEMA
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
    compare = manifest["schema_version"] != SCHEMA
    if (manifest["schema_version"] == PARTICLE_SCHEMA) != ("particles" in settings):
        raise ValueError("Particle benchmark settings differ")
    expected_arrays, expected_sheets = benchmark_results(settings, compare=compare)
    if compare:
        timing = sheets.get("Timing")
        pairs = [(case, method) for case in CASES for method in benchmark_methods(settings, compare)]
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
