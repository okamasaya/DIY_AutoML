"""First-order extended Kalman filtering with additive Gaussian noise."""

from threading import Lock

import numpy as np

from .state_space import check_covariance, check_initial_mean, finite_array, gaussian_filter_inputs, symmetric


_RESULT_FIELDS = (
    "forecast", "forecast_covariance", "filtered", "filtered_covariance",
    "predicted_observation", "predicted_observation_covariance",
)


def _filter_step(observed, mean, covariance, process, noise, step, model, identity, control):
    transition, transition_jacobian, measurement, measurement_jacobian = model
    transition_args = (mean, step) if control is None else (mean, step, control)
    predicted = finite_array(transition(*transition_args), mean.shape, "Transition result")
    f_jacobian = finite_array(transition_jacobian(*transition_args), covariance.shape, "Transition Jacobian")
    prediction_covariance = symmetric(f_jacobian @ covariance @ f_jacobian.T + process)
    expected = finite_array(measurement(predicted, step), observed.shape, "Measurement result")
    h_jacobian = finite_array(
        measurement_jacobian(predicted, step), (observed.size, mean.size), "Measurement Jacobian"
    )
    cross = prediction_covariance @ h_jacobian.T
    innovation_covariance = symmetric(h_jacobian @ cross + noise)

    innovation = observed - expected
    available = ~np.isnan(observed)
    mean, covariance = predicted, prediction_covariance
    if available.any():
        update_covariance = innovation_covariance
        if not available.all():
            selection = np.ix_(available, available)
            update_covariance = innovation_covariance[selection]
            noise = noise[selection]
            h_jacobian = h_jacobian[available]
            cross = cross[:, available]
            innovation = innovation[available]
        gain = np.linalg.solve(update_covariance, cross.T).T
        mean = predicted + gain @ innovation
        residual = identity - gain @ h_jacobian
        # Joseph form for the covariance update with the observed components.
        covariance = symmetric(residual @ prediction_covariance @ residual.T + gain @ noise @ gain.T)
    return predicted, prediction_covariance, mean, covariance, expected, innovation_covariance


class ExtendedKalmanFilter:
    """Maintain one EKF state for observations that arrive in time order.

    The initial mean and covariance describe step 0. The process and observation
    covariances are fixed matrices and are validated at construction. Model
    callbacks receive (state, step), with step starting at 1. If a step supplies
    a control, the transition and its Jacobian receive (state, step, control).
    Measurement callbacks keep the two-argument signature.
    """

    def __init__(self, *, initial_mean, initial_covariance, transition, transition_jacobian,
                 measurement, measurement_jacobian, process_covariance, observation_covariance):
        mean = check_initial_mean(initial_mean)
        noise = np.asarray(observation_covariance, dtype=float)
        if noise.ndim != 2 or not noise.size:
            raise ValueError("Observation covariance must be a nonempty matrix")
        self._mean = mean.copy()
        self._covariance = check_covariance(initial_covariance, mean.size, "Initial covariance")
        self._process = check_covariance(process_covariance, mean.size, "Process covariance")
        self._noise = check_covariance(noise, noise.shape[0], "Observation covariance", positive=True)
        self._model = transition, transition_jacobian, measurement, measurement_jacobian
        self._identity = np.eye(mean.size)
        self._observation_shape = (noise.shape[0],)
        self._step = 0
        self._lock = Lock()

    def step(self, observation, *, control=None):
        """Process one observation vector, optionally with a finite scalar or vector control.

        NaN skips an observation component. The returned arrays have the same keys
        as the batch result without its leading time axis. The state advances only
        after a successful update. Submit observations in chronological order.
        """
        observed = np.asarray(observation, dtype=float)
        if observed.shape != self._observation_shape or np.isinf(observed).any():
            raise ValueError(f"Observation must have shape {self._observation_shape} and contain finite values or NaN")
        if control is not None:
            control = np.asarray(control, dtype=float)
            if control.ndim > 1 or not control.size or not np.isfinite(control).all():
                raise ValueError("Control must be a finite scalar or nonempty vector")
            control = control[()]

        with self._lock, np.errstate(over="raise", invalid="raise", divide="raise"):
            step = self._step + 1
            result = _filter_step(
                observed, self._mean.copy(), self._covariance, self._process, self._noise,
                step, self._model, self._identity, control,
            )
            self._mean, self._covariance = result[2].copy(), result[3].copy()
            self._step = step
        return dict(zip(_RESULT_FIELDS, result, strict=True))


def extended_kalman_filter(observations, *, initial_mean, initial_covariance, transition, transition_jacobian,
                           measurement, measurement_jacobian, process_covariance, observation_covariance,
                           controls=None):
    """Filter one sequence with known nonlinear models and their analytic Jacobians.

    The initial mean and covariance describe step 0. Observation rows correspond
    to steps 1 onward. Model callbacks receive (state, step), where step is the
    destination for a transition. With controls, the transition and its Jacobian
    receive (state, step, control). Supply controls with shape (steps,) for scalar
    controls or (steps, control_dimensions) for vector controls. Covariances Q and
    R can be fixed matrices or one matrix per observation row. NaN observation
    components are skipped.
    """
    observations, mean, covariance, process, noise = gaussian_filter_inputs(
        observations, initial_mean, initial_covariance, process_covariance, observation_covariance,
    )
    steps, sensors = observations.shape
    states = mean.size
    if controls is not None:
        controls = np.asarray(controls, dtype=float)
        if (controls.ndim not in (1, 2) or len(controls) != steps
                or not controls.size or not np.isfinite(controls).all()):
            raise ValueError("Controls must contain one finite scalar or nonempty vector per observation row")

    shapes = ((states,), (states, states), (states,), (states, states), (sensors,), (sensors, sensors))
    outputs = tuple(np.empty((steps, *shape)) for shape in shapes)
    model = transition, transition_jacobian, measurement, measurement_jacobian
    identity = np.eye(states)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for index, observed in enumerate(observations):
            control = None if controls is None else controls[index]
            result = _filter_step(
                observed, mean, covariance, process[index], noise[index], index + 1, model, identity, control,
            )
            for output, value in zip(outputs, result, strict=True):
                output[index] = value
            mean, covariance = result[2:4]
    return dict(zip(_RESULT_FIELDS, outputs, strict=True))
