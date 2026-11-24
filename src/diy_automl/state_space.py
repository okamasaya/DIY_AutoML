"""Array validation shared by the Gaussian state-space filters."""

import numpy as np


def finite_array(value, shape, name):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return array


def symmetric(matrix):
    return 0.5 * matrix + 0.5 * matrix.swapaxes(-1, -2)


def eigenvalue_tolerance(eigenvalues):
    dimension = eigenvalues.shape[-1]
    return dimension * np.finfo(float).eps * np.max(np.abs(eigenvalues), axis=-1)


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
        if np.any(eigenvalues[..., 0] < -eigenvalue_tolerance(eigenvalues)):
            raise ValueError(f"{name} must be positive semidefinite")
    return np.broadcast_to(value, (steps, dimension, dimension)) if steps is not None else value


def gaussian_filter_inputs(observations, initial_mean, initial_covariance, process_covariance, observation_covariance):
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
    return observations, mean, covariance, process, noise
