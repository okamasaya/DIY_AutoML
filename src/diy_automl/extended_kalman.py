"""First-order extended Kalman filtering with additive Gaussian noise."""

import numpy as np

from .state_space import finite_array, gaussian_filter_inputs, symmetric


def extended_kalman_filter(observations, *, initial_mean, initial_covariance, transition, transition_jacobian,
                           measurement, measurement_jacobian, process_covariance, observation_covariance):
    """Filter one sequence with known nonlinear models and their analytic Jacobians.

    The initial mean and covariance describe step 0. Observation rows correspond
    to steps 1 onward. Model callbacks receive (state, step), where step is the
    destination for a transition. Covariances Q and R can be fixed matrices or one
    matrix per observation row. NaN observation components are skipped.
    """
    observations, mean, covariance, process, noise = gaussian_filter_inputs(
        observations, initial_mean, initial_covariance, process_covariance, observation_covariance,
    )
    steps, sensors = observations.shape
    states = len(mean)

    prior = np.empty((steps, states))
    posterior = np.empty_like(prior)
    prior_covariance = np.empty((steps, states, states))
    posterior_covariance = np.empty_like(prior_covariance)
    predicted_observation = np.empty((steps, sensors))
    predicted_observation_covariance = np.empty((steps, sensors, sensors))
    identity = np.eye(states)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for index, observed in enumerate(observations):
            step = index + 1
            predicted = finite_array(transition(mean, step), (states,), "Transition result")
            f_jacobian = finite_array(transition_jacobian(mean, step), (states, states), "Transition Jacobian")
            prediction_covariance = symmetric(f_jacobian @ covariance @ f_jacobian.T + process[index])
            expected = finite_array(measurement(predicted, step), (sensors,), "Measurement result")
            h_jacobian = finite_array(measurement_jacobian(predicted, step), (sensors, states), "Measurement Jacobian")
            cross = prediction_covariance @ h_jacobian.T
            innovation_covariance = symmetric(h_jacobian @ cross + noise[index])

            prior[index], prior_covariance[index] = predicted, prediction_covariance
            predicted_observation[index], predicted_observation_covariance[index] = expected, innovation_covariance
            available = np.flatnonzero(np.isfinite(observed))
            mean, covariance = predicted, prediction_covariance
            if available.size:
                selection = np.ix_(available, available)
                gain = np.linalg.solve(innovation_covariance[selection], cross[:, available].T).T
                mean = predicted + gain @ (observed[available] - expected[available])
                residual = identity - gain @ h_jacobian[available]
                # Joseph form for the covariance update with the observed components.
                covariance = symmetric(residual @ prediction_covariance @ residual.T + gain @ noise[index][selection] @ gain.T)
            posterior[index], posterior_covariance[index] = mean, covariance

    return {"forecast": prior, "forecast_covariance": prior_covariance,
            "filtered": posterior, "filtered_covariance": posterior_covariance,
            "predicted_observation": predicted_observation,
            "predicted_observation_covariance": predicted_observation_covariance}
