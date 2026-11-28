"""Compare EKF estimates and reported uncertainty under known model mismatches."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import chi2, norm

import uc_automail as uc_al


SEED = 20261128
TRAJECTORIES = 16
BURN_IN = 64
EVALUATION_STEPS = 384
DT = 0.1
CONFIDENCE = 0.95

TRANSITION = np.array([[1, DT], [0, 1]])
INPUT = np.array([DT**2 / 2, DT])
MEASUREMENT = np.array([[1, 0]])
INITIAL_COVARIANCE = np.diag([0.04, 0.04])
PROCESS_COVARIANCE = 0.09 * np.array([[DT**3 / 3, DT**2 / 2], [DT**2 / 2, DT]])
OBSERVATION_COVARIANCE = np.array([[0.1**2]])


def generate_data():
    rng = np.random.default_rng(SEED)
    steps = BURN_IN + EVALUATION_STEPS
    controls = 0.6 * np.sin(np.arange(steps) * DT)
    previous = rng.multivariate_normal(np.zeros(2), INITIAL_COVARIANCE, size=TRAJECTORIES)
    disturbances = rng.multivariate_normal(np.zeros(2), PROCESS_COVARIANCE, size=(steps, TRAJECTORIES))
    truth = np.empty((TRAJECTORIES, steps, 2))
    for index, control in enumerate(controls):
        previous = previous @ TRANSITION.T + INPUT * control + disturbances[index]
        truth[:, index] = previous
    noise = rng.normal(0, np.sqrt(OBSERVATION_COVARIANCE[0, 0]), size=(TRAJECTORIES, steps, 1))
    return controls, truth, truth @ MEASUREMENT.T + noise


def evaluate(controls, truth, observations, *, q_scale, r_scale, position_bias):
    nis, nees, pull, errors, position_sd, lag1, pvalues = [], [], [], [], [], [], []
    window = slice(BURN_IN, None)
    for reference, observed in zip(truth, observations, strict=True):
        observed = observed + position_bias
        filtered = uc_al.extended_kalman_filter(
            observed, initial_mean=[0, 0], initial_covariance=INITIAL_COVARIANCE,
            transition=lambda state, step, control: TRANSITION @ state + INPUT * control,
            transition_jacobian=lambda state, step, control: TRANSITION,
            measurement=lambda state, step: MEASUREMENT @ state,
            measurement_jacobian=lambda state, step: MEASUREMENT,
            process_covariance=PROCESS_COVARIANCE * q_scale,
            observation_covariance=OBSERVATION_COVARIANCE * r_scale, controls=controls,
        )
        evaluation = {key: values[window] for key, values in filtered.items()}
        diagnostics = uc_al.filter_diagnostics(evaluation, observed[window], confidence=CONFIDENCE)
        consistency = uc_al.evaluate_filter_consistency(reference[window], evaluation, confidence=CONFIDENCE)
        nis.append(diagnostics["nis"])
        nees.append(consistency["nees"])
        pull.append(consistency["pull"][:, 0])
        errors.append(consistency["error"][:, 0])
        position_sd.append(np.sqrt(evaluation["filtered_covariance"][:, 0, 0]))
        lag1.append(diagnostics["autocorrelation"][0, 0])
        pvalues.append(diagnostics["ljung_box_pvalue"][0])
    position_limit = norm.isf((1 - CONFIDENCE) / 2)
    state_limit = chi2.ppf(CONFIDENCE, 2)
    return {
        "q_scale": q_scale,
        "r_scale": r_scale,
        "position_observation_bias_m": position_bias,
        "position_rmse_m": float(np.sqrt(np.mean(np.square(errors)))),
        "position_mean_error_m": float(np.mean(errors)),
        "position_coverage": float(np.mean(np.abs(pull) <= position_limit)),
        "mean_position_interval_width_m": float(2 * position_limit * np.mean(position_sd)),
        "state_ellipse_coverage": float(np.mean(np.asarray(nees) <= state_limit)),
        "mean_nis_per_dimension": float(np.mean(nis)),
        "mean_nees_per_dimension": float(np.mean(nees) / 2),
        "mean_innovation_lag1": float(np.mean(lag1)),
        "ljung_box_rejected_trajectories": int(np.count_nonzero(np.asarray(pvalues) < 1 - CONFIDENCE)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = generate_data()
    conditions = (
        ("matched", 1.0, 1.0, 0.0),
        ("q_quarter", 0.25, 1.0, 0.0),
        ("q_fourfold", 4.0, 1.0, 0.0),
        ("r_quarter", 1.0, 0.25, 0.0),
        ("r_fourfold", 1.0, 4.0, 0.0),
        ("position_observation_bias", 1.0, 1.0, 0.15),
    )
    result = {
        "seed": SEED, "trajectories": TRAJECTORIES, "warmup_steps": BURN_IN,
        "evaluation_steps_per_trajectory": EVALUATION_STEPS, "step_s": DT,
        "confidence": CONFIDENCE, "numpy_version": np.__version__, "package_version": uc_al.__version__,
        "true_process_covariance": PROCESS_COVARIANCE.tolist(),
        "true_observation_covariance": OBSERVATION_COVARIANCE.tolist(),
        "conditions": {
            name: evaluate(*data, q_scale=q, r_scale=r, position_bias=bias)
            for name, q, r, bias in conditions
        },
    }
    contents = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.write_text(contents, encoding="utf-8")
    else:
        print(contents, end="")


if __name__ == "__main__":
    main()
