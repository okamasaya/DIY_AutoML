"""First-order extended Kalman filtering with additive Gaussian noise."""

import numpy as np


def finite_array(value, shape, name):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return array


def symmetric(matrix):
    return 0.5 * matrix + 0.5 * matrix.swapaxes(-1, -2)


def check_covariance(value, dimension, name, *, steps=None, positive=False):
    value = np.asarray(value, dtype=float)
    shape = (dimension, dimension)
    if steps is not None and value.shape == (steps, *shape):
        shape = value.shape
    value = finite_array(value, shape, name)
    if not np.allclose(value, value.swapaxes(-1, -2), rtol=1e-12, atol=0):
        raise ValueError(f"{name} must be symmetric")
    value = symmetric(value)
    if positive:
        try:
            np.linalg.cholesky(value)
        except np.linalg.LinAlgError as error:
            raise ValueError(f"{name} must be positive definite") from error
    else:
        eigenvalues = np.linalg.eigvalsh(value)
        tolerance = dimension * np.finfo(float).eps * np.max(np.abs(eigenvalues), axis=-1)
        if np.any(eigenvalues[..., 0] < -tolerance):
            raise ValueError(f"{name} must be positive semidefinite")
    return np.broadcast_to(value, (steps, dimension, dimension)) if steps is not None else value


def extended_kalman_filter(observations, *, initial_mean, initial_covariance, transition, transition_jacobian,
                           measurement, measurement_jacobian, process_covariance, observation_covariance):
    """Filter one sequence with known nonlinear models and their analytic Jacobians.

    The initial mean and covariance describe step 0. Observation rows correspond
    to steps 1 onward. Model callbacks receive (state, step), where step is the
    destination for a transition. Covariances Q and R can be fixed matrices or one
    matrix per observation row. NaN observation components are skipped.
    """
    observations = np.asarray(observations, dtype=float)
    mean = np.asarray(initial_mean, dtype=float)
    if mean.ndim != 1 or not mean.size or not np.isfinite(mean).all():
        raise ValueError("Initial mean must be a nonempty finite state vector")
    if observations.ndim != 2 or not observations.size or np.isinf(observations).any():
        raise ValueError("Observations must be a nonempty (steps, measurements) array of finite values or NaN")
    steps, sensors = observations.shape
    states = len(mean)
    covariance = check_covariance(initial_covariance, states, "Initial covariance")
    process = check_covariance(process_covariance, states, "Process covariance", steps=steps)
    noise = check_covariance(observation_covariance, sensors, "Observation covariance", steps=steps, positive=True)

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
