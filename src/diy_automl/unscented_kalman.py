"""Scaled unscented Kalman filtering for additive Gaussian noise."""

import numpy as np

from .state_space import covariance_factor, finite_array, gaussian_filter_inputs, symmetric


def sigma_points(mean, covariance):
    offsets = np.sqrt(len(mean)) * covariance_factor(covariance).T
    return np.concatenate([mean[None, :], mean + offsets, mean - offsets])


def transform(points, function, step, dimension, name, mean_weights):
    values = np.stack([finite_array(function(point, step), (dimension,), name) for point in points])
    center = values[0]
    mean = center + mean_weights @ (values - center)
    return mean, values - mean


def unscented_kalman_filter(observations, *, initial_mean, initial_covariance, transition, measurement,
                            process_covariance, observation_covariance):
    """Filter one sequence with symmetric sigma points and additive noise.

    The initial state describes step 0. Callbacks receive (state, step) with steps
    starting at 1. Q and R can be fixed or specified per row. NaN observation
    components are skipped. The scaled transform fixes alpha=1, beta=2, kappa=0.
    """
    observations, mean, covariance, process, noise = gaussian_filter_inputs(
        observations, initial_mean, initial_covariance, process_covariance, observation_covariance,
    )
    steps, sensors = observations.shape
    states = len(mean)
    mean_weights = np.full(2 * states + 1, 1 / (2 * states))
    mean_weights[0] = 0
    covariance_weights = mean_weights[:, None].copy()
    covariance_weights[0, 0] = 2

    prior = np.empty((steps, states))
    posterior = np.empty_like(prior)
    prior_covariance = np.empty((steps, states, states))
    posterior_covariance = np.empty_like(prior_covariance)
    predicted_observation = np.empty((steps, sensors))
    predicted_observation_covariance = np.empty((steps, sensors, sensors))
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for index, observed in enumerate(observations):
            step = index + 1
            predicted, state_residuals = transform(sigma_points(mean, covariance), transition, step,
                                                   states, "Transition result", mean_weights)
            prediction_covariance = symmetric(state_residuals.T @ (covariance_weights * state_residuals) + process[index])

            # Regenerate points after adding Q so measurement moments include process noise.
            points = sigma_points(predicted, prediction_covariance)
            expected, observation_residuals = transform(points, measurement, step, sensors,
                                                         "Measurement result", mean_weights)
            weighted_observation_residuals = covariance_weights * observation_residuals
            innovation_covariance = symmetric(observation_residuals.T @ weighted_observation_residuals + noise[index])
            prior[index], prior_covariance[index] = predicted, prediction_covariance
            predicted_observation[index], predicted_observation_covariance[index] = expected, innovation_covariance

            available = np.flatnonzero(np.isfinite(observed))
            mean, covariance = predicted, prediction_covariance
            if available.size:
                selection = np.ix_(available, available)
                state_residuals = points - predicted
                cross = state_residuals.T @ weighted_observation_residuals[:, available]
                gain = np.linalg.solve(innovation_covariance[selection], cross.T).T
                mean = predicted + gain @ (observed[available] - expected[available])
                # Sum weighted residual outer products and measurement noise without covariance subtraction.
                residuals = state_residuals - observation_residuals[:, available] @ gain.T
                covariance = symmetric(residuals.T @ (covariance_weights * residuals) + gain @ noise[index][selection] @ gain.T)
            posterior[index], posterior_covariance[index] = mean, covariance

    return {"forecast": prior, "forecast_covariance": prior_covariance,
            "filtered": posterior, "filtered_covariance": posterior_covariance,
            "predicted_observation": predicted_observation,
            "predicted_observation_covariance": predicted_observation_covariance}
