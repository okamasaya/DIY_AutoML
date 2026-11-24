"""Generate and check a synthetic Voyager-inspired tracking dataset."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from diy_automl import extended_kalman_filter, unscented_kalman_filter


AU_KM = 149_597_870.7
DAYS_PER_YEAR = 365.25
SUN_MU = (2 * np.pi) ** 2  # AU^3 per year^2, normalized to the circular Earth orbit.
TIME_DAYS = np.arange(0, 729, 7)
STEP = 7 / DAYS_PER_YEAR
EARTH_PHASE = 1.0
TRUE_START = np.array([9.5, 0.0, 4.35, 1.0])
PRIOR_MEAN = np.array([9.502, -0.004, 4.362, 0.985])
PRIOR_STD = np.array([0.005, 0.01, 0.02, 0.03])
PROCESS_STD = np.array([1e-6, 1e-6, 2e-5, 2e-5])
OBSERVATION_STD = np.array([2e-4, 2e-3])
STATE_COLUMNS = ("x_au", "y_au", "vx_au_per_year", "vy_au_per_year")
OBSERVATION_COLUMNS = ("range_au", "range_rate_au_per_year")
FULL_GAP = (266, 322)
RATE_GAP = (112, 126)
SPIKE_DAYS = (175, 399, 560)
SPIKE_SIGMA = np.array([[10, -6], [-12, 8], [8, -10]])
BIAS_START_DAY = 420
BIAS_SIGMA = np.array([6, -4])


def derivative(state):
    position, velocity = state[:2], state[2:]
    return np.r_[velocity, -SUN_MU * position / np.linalg.norm(position) ** 3]


def transition(state, _step):
    first = derivative(state)
    second = derivative(state + STEP * first / 2)
    third = derivative(state + STEP * second / 2)
    fourth = derivative(state + STEP * third)
    return state + STEP * (first + 2 * second + 2 * third + fourth) / 6


def transition_jacobian(state, step):
    shifts = np.eye(len(state)) * 1e-5
    return np.column_stack([(transition(state + d, step) - transition(state - d, step)) / 2e-5 for d in shifts])


def earth_states(time_years):
    phase = EARTH_PHASE + 2 * np.pi * time_years
    x, y = np.cos(phase), np.sin(phase)
    return np.column_stack((x, y, -2 * np.pi * y, 2 * np.pi * x))


def measurement(state, observer):
    position, velocity = np.split(state - observer, 2)
    distance = np.linalg.norm(position)
    return np.array([distance, position @ velocity / distance])


def measurement_jacobian(state, observer):
    position, velocity = np.split(state - observer, 2)
    distance = np.linalg.norm(position)
    direction = position / distance
    transverse_velocity = velocity - (direction @ velocity) * direction
    return np.array([np.r_[direction, 0, 0], np.r_[transverse_velocity / distance, direction]])


def make_samples(seed):
    time = TIME_DAYS / DAYS_PER_YEAR
    solution = solve_ivp(lambda _time, state: derivative(state), (0, time[-1]), TRUE_START,
                         t_eval=time, method="DOP853", rtol=1e-12, atol=1e-13)
    if not solution.success:
        raise RuntimeError(solution.message)
    truth = solution.y.T
    days = TIME_DAYS[1:]
    earth = earth_states(time[1:])
    ideal = np.array([measurement(state, observer) for state, observer in zip(truth[1:], earth, strict=True)])
    noise = np.random.default_rng(seed).normal(0, OBSERVATION_STD, ideal.shape)
    nominal = ideal + noise
    scenarios = {name: nominal.copy() for name in ("nominal", "missing", "outliers", "bias")}
    scenarios["missing"][(FULL_GAP[0] <= days) & (days <= FULL_GAP[1])] = np.nan
    scenarios["missing"][(RATE_GAP[0] <= days) & (days <= RATE_GAP[1]), 1] = np.nan
    scenarios["outliers"][np.searchsorted(days, SPIKE_DAYS)] += SPIKE_SIGMA * OBSERVATION_STD
    scenarios["bias"][days >= BIAS_START_DAY] += BIAS_SIGMA * OBSERVATION_STD
    return truth, earth, scenarios


def score_track(estimate, truth):
    error = estimate - truth[1:]
    position_error = np.linalg.norm(error[:, :2], axis=1) * AU_KM
    speed_error = np.linalg.norm(error[:, 2:], axis=1) * AU_KM * 1_000 / (DAYS_PER_YEAR * 86_400)
    return {
        "position_rmse_km": float(np.sqrt(np.mean(position_error ** 2))),
        "velocity_rmse_m_per_s": float(np.sqrt(np.mean(speed_error ** 2))),
    }, position_error


def compare_filters(truth, earth, scenarios):
    config = dict(
        initial_mean=PRIOR_MEAN,
        initial_covariance=np.diag(PRIOR_STD ** 2),
        transition=transition,
        measurement=lambda state, step: measurement(state, earth[step - 1]),
        process_covariance=np.diag(PROCESS_STD ** 2),
        observation_covariance=np.diag(OBSERVATION_STD ** 2),
    )
    open_loop = [PRIOR_MEAN]
    for step in range(1, len(TIME_DAYS)):
        open_loop.append(transition(open_loop[-1], step))
    open_loop_metrics, _ = score_track(np.array(open_loop[1:]), truth)
    scores, ekf_errors = {}, {}
    for name, observations in scenarios.items():
        ekf = extended_kalman_filter(
            observations, **config, transition_jacobian=transition_jacobian,
            measurement_jacobian=lambda state, step: measurement_jacobian(state, earth[step - 1]),
        )
        ukf = unscented_kalman_filter(observations, **config)
        filtered_score, ekf_errors[name] = score_track(ekf["filtered"], truth)
        scores[name] = {
            "missing_observations": np.isnan(observations).sum(axis=0).tolist(),
            "EKF": {"forecast": score_track(ekf["forecast"], truth)[0], "filtered": filtered_score},
            "UKF": {stage: score_track(ukf[stage], truth)[0] for stage in ("forecast", "filtered")},
        }
    return open_loop_metrics, scores, ekf_errors


def plot_track(path, truth, errors):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(11, 4.8), layout="constrained")
    FigureCanvasAgg(figure)
    orbit, error = figure.subplots(1, 2, gridspec_kw={"width_ratios": [1, 1.35]})
    colors = ("#0072B2", "#009E73", "#E69F00", "#CC79A7")
    earth = earth_states(np.linspace(0, 1, 150))
    orbit.plot(earth[:, 0], earth[:, 1], color="#607485", label="Earth orbit")
    orbit.scatter(0, 0, color="#e5a319", label="Sun", s=35, zorder=3)
    orbit.plot(truth[:, 0], truth[:, 1], color=colors[0], label="Probe truth")
    orbit.scatter(truth[[0, -1], 0], truth[[0, -1], 1], color=colors[0], s=20, zorder=3)
    orbit.set(title="Heliocentric trajectory", xlabel="x (AU)", ylabel="y (AU)", aspect="equal")
    orbit.set_xlim(-2, 19)
    orbit.set_ylim(-3, 6)
    orbit.legend(loc="upper center", ncols=3, frameon=False, fontsize=8)
    for (name, values), color in zip(errors.items(), colors, strict=True):
        error.plot(TIME_DAYS[1:], values, label=name.capitalize(), color=color)
    error.set(title="EKF position error after each step", xlabel="Elapsed days", ylabel="Position error (km)")
    error.legend(frameon=False, ncols=2)
    for axis in (orbit, error):
        axis.grid(alpha=0.22)
    figure.suptitle("Voyager-inspired synthetic tracking sample", fontsize=14)
    figure.savefig(path, dpi=160, facecolor="white")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    truth, earth, scenarios = make_samples(args.seed)
    open_loop, scores, errors = compare_filters(truth, earth, scenarios)
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "truth.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time_day", *STATE_COLUMNS))
        writer.writerows((day, *state) for day, state in zip(TIME_DAYS, truth, strict=True))
    with (args.output / "observations.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("scenario", "time_day", *(f"earth_{field}" for field in STATE_COLUMNS), *OBSERVATION_COLUMNS))
        for name, observations in scenarios.items():
            for day, observer, reading in zip(TIME_DAYS[1:], earth, observations, strict=True):
                writer.writerow((name, day, *observer, *(None if np.isnan(value) else value for value in reading)))
    report = {
        "dataset": "synthetic Voyager-inspired outbound arc",
        "seed": args.seed,
        "days_per_year": DAYS_PER_YEAR,
        "solar_mu_au3_per_year2": SUN_MU,
        "step_days": int(TIME_DAYS[1]),
        "earth_initial_phase_rad": EARTH_PHASE,
        "initial_prior_mean": PRIOR_MEAN.tolist(),
        "initial_prior_sd": PRIOR_STD.tolist(),
        "process_sd_per_step": PROCESS_STD.tolist(),
        "observation_sd_by_column": dict(zip(OBSERVATION_COLUMNS, OBSERVATION_STD.tolist(), strict=True)),
        "scenario_events": {
            "missing": {"both_channels_days_inclusive": FULL_GAP, "rate_only_days_inclusive": RATE_GAP},
            "outliers": {"days": SPIKE_DAYS, "added_sd_by_channel": SPIKE_SIGMA.tolist()},
            "bias": {"from_day_inclusive": BIAS_START_DAY, "added_sd_by_channel": BIAS_SIGMA.tolist()},
        },
        "open_loop": open_loop,
        "scores": scores,
    }
    with (args.output / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    plot_track(args.output / "trajectory.png", truth, errors)
    print(args.output)


if __name__ == "__main__":
    main()
