"""Bootstrap particle filtering with weighted summaries and systematic resampling."""

from numbers import Integral, Real

import numpy as np
from scipy.linalg import solve_triangular

from .state_space import covariance_factor, finite_array, gaussian_filter_inputs, symmetric


QUANTILES = (0.025, 0.975)


def cumulative_weights(weights):
    cumulative = np.cumsum(weights)
    return cumulative / cumulative[-1]


def weighted_bounds(samples, weights):
    order = np.argsort(samples, axis=0, kind="stable")
    sorted_samples = np.take_along_axis(samples, order, axis=0)
    cumulative = np.cumsum(weights[order], axis=0)
    cumulative /= cumulative[-1]
    lower = np.argmax(cumulative >= QUANTILES[0], axis=0)
    upper = np.argmax(cumulative >= QUANTILES[1], axis=0)
    dimensions = np.arange(samples.shape[1])
    return sorted_samples[lower, dimensions], sorted_samples[upper, dimensions]


def moments(samples, weights):
    mean = weights @ samples
    residuals = samples - mean
    return mean, symmetric(residuals.T @ (weights[:, None] * residuals))


def save_moments(trace, name, index, samples, weights, interval_samples=None, noise=None):
    mean, covariance = moments(samples, weights)
    if noise is not None:
        covariance = covariance + noise
    lower, upper = weighted_bounds(samples if interval_samples is None else interval_samples, weights)
    trace[name][index] = mean
    trace[f"{name}_covariance"][index] = covariance
    trace[f"{name}_lower"][index] = lower
    trace[f"{name}_upper"][index] = upper


def bootstrap_particle_filter(observations, *, initial_mean, initial_covariance, transition, measurement,
                              process_covariance, observation_covariance, particles=256, seed=42,
                              resample_fraction=0.5, record_steps=()):
    """Filter a sequence with Gaussian initial, process and observation noise.

    Callbacks receive an array of shape (particles, states) and a 1-based step.
    They return one row per particle. Q and R can be fixed or per observation.
    NaN observation components are skipped. Requested steps retain a weighted
    state sample before resampling; other steps store only summaries.
    """
    observations, mean, covariance, process, noise = gaussian_filter_inputs(
        observations, initial_mean, initial_covariance, process_covariance, observation_covariance,
    )
    steps, sensors = observations.shape
    states = len(mean)
    if isinstance(particles, bool) or not isinstance(particles, Integral) or particles < 2:
        raise ValueError("Use at least two particles")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("Seed must be a nonnegative integer")
    if not isinstance(resample_fraction, Real) or not 0 <= resample_fraction <= 1:
        raise ValueError("Resampling fraction must be from 0 to 1")
    record_steps = tuple(record_steps)
    if any(isinstance(step, bool) or not isinstance(step, Integral) or not 1 <= step <= steps for step in record_steps):
        raise ValueError("Recorded steps must be 1-based observation indices")
    recorded = {step: index for index, step in enumerate(sorted(set(record_steps)))}
    rng, interval_rng = (np.random.default_rng(child) for child in np.random.SeedSequence(int(seed)).spawn(2))
    cloud = mean + rng.standard_normal((particles, states)) @ covariance_factor(covariance).T
    uniform_log_weight = -np.log(particles)
    log_weights = np.full(particles, uniform_log_weight)
    weights = np.full(particles, 1 / particles)
    trace = {}
    for name, dimension in (("forecast", states), ("filtered", states), ("predicted_observation", sensors)):
        trace[name] = np.empty((steps, dimension))
        trace[f"{name}_covariance"] = np.empty((steps, dimension, dimension))
        trace[f"{name}_lower"] = np.empty((steps, dimension))
        trace[f"{name}_upper"] = np.empty((steps, dimension))
    trace["effective_sample_size"] = np.empty(steps)
    trace["resampled"] = np.zeros(steps, dtype=bool)
    trace["filtered_negative_mass"] = np.empty((steps, states))
    if recorded:
        trace["recorded_steps"] = np.array(list(recorded))
        trace["recorded_particles"] = np.empty((len(recorded), particles, states))
        trace["recorded_weights"] = np.empty((len(recorded), particles))

    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        for index, observed in enumerate(observations):
            step = index + 1
            predicted = finite_array(transition(cloud, step), (particles, states), "Transition result")
            cloud = predicted + rng.standard_normal(predicted.shape) @ covariance_factor(process[index]).T
            projected = finite_array(measurement(cloud, step), (particles, sensors), "Measurement result")
            predictive_readings = projected + interval_rng.standard_normal(projected.shape) @ np.linalg.cholesky(noise[index]).T
            save_moments(trace, "forecast", index, cloud, weights)
            save_moments(trace, "predicted_observation", index, projected, weights, predictive_readings, noise[index])

            available = np.flatnonzero(np.isfinite(observed))
            if available.size:
                selected_noise = noise[index][np.ix_(available, available)]
                residual = observed[available] - projected[:, available]
                if available.size == 1:
                    standardized = residual / np.sqrt(selected_noise[0, 0])
                else:
                    standardized = solve_triangular(np.linalg.cholesky(selected_noise), residual.T,
                                                    lower=True, check_finite=False).T
                log_weights = log_weights - 0.5 * np.sum(standardized**2, axis=1)
                log_weights -= np.max(log_weights)
                weights = np.exp(log_weights)
                total = weights.sum()
                log_weights -= np.log(total)
                weights /= total
                save_moments(trace, "filtered", index, cloud, weights)
            else:
                for suffix in ("", "_covariance", "_lower", "_upper"):
                    trace[f"filtered{suffix}"][index] = trace[f"forecast{suffix}"][index]

            effective = min(particles, 1 / (weights @ weights))
            trace["effective_sample_size"][index] = effective
            trace["filtered_negative_mass"][index] = weights @ (cloud < 0)
            if step in recorded:
                slot = recorded[step]
                trace["recorded_particles"][slot], trace["recorded_weights"][slot] = cloud, weights
            if available.size and effective < resample_fraction * particles:
                positions = (rng.random() + np.arange(particles)) / particles
                positions = np.minimum(positions, np.nextafter(1.0, 0.0))
                indices = np.searchsorted(cumulative_weights(weights), positions, side="right")
                cloud = cloud[indices]
                weights.fill(1 / particles)
                log_weights.fill(uniform_log_weight)
                trace["resampled"][index] = True

    return trace
