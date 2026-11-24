"""Reproduce one-step forecasts on a small UCI Bike Sharing daily sample."""

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from diy_automl import bootstrap_particle_filter, extended_kalman_filter
from diy_automl.kalman_workflow import interval


def check_bike_sample(archive, particles):
    with ZipFile(archive) as source:
        data = source.read("day.csv")
    daily = pd.read_csv(BytesIO(data), usecols=["dteday", "cnt"], parse_dates=["dteday"])
    daily = daily.sort_values("dteday").head(150)
    dates = daily["dteday"]
    counts = daily["cnt"].to_numpy(dtype=float)
    if len(daily) != 150 or not dates.diff().iloc[1:].eq(pd.Timedelta(days=1)).all() or not np.isfinite(counts).all():
        raise ValueError("The sample needs 150 consecutive days with finite rental counts")

    # Specify a local-level model from the calibration prefix only.
    calibration_days = 60
    change_variance = float(np.mean(np.diff(counts[:calibration_days])**2))
    process_variance, observation_variance = 0.1 * change_variance, 0.45 * change_variance
    readings = counts[1:, None]
    settings = dict(initial_mean=[counts[0]], initial_covariance=[[observation_variance]],
                    transition=lambda x, k: x, measurement=lambda x, k: x,
                    process_covariance=[[process_variance]], observation_covariance=[[observation_variance]])
    kalman = extended_kalman_filter(readings, **settings, transition_jacobian=lambda x, k: [[1]],
                                    measurement_jacobian=lambda x, k: [[1]])
    particle = bootstrap_particle_filter(readings, **settings, particles=particles, seed=42)

    held = slice(calibration_days - 1, None)
    truth = readings[held, 0]
    kalman_bounds = interval(kalman["predicted_observation"][held, 0],
                             kalman["predicted_observation_covariance"][held, 0, 0])
    particle_bounds = (particle["predicted_observation_lower"][held, 0], particle["predicted_observation_upper"][held, 0])
    metrics = []
    for name, forecast, bounds in (
        ("Last observation", counts[:-1][held], None),
        ("Linear Kalman", kalman["predicted_observation"][held, 0], kalman_bounds),
        ("Particle filter", particle["predicted_observation"][held, 0], particle_bounds),
    ):
        row = dict(method=name, rmse=float(np.sqrt(np.mean((forecast - truth)**2))))
        if bounds is not None:
            lower, upper = bounds
            row.update(coverage=float(np.mean((lower <= truth) & (truth <= upper))),
                       mean_interval_width=float(np.mean(upper - lower)))
        metrics.append(row)
    return {
        "day_csv_sha256": hashlib.sha256(data).hexdigest(), "particles": particles, "particle_seed": 42,
        "calibration_days": calibration_days, "evaluation_days": len(truth),
        "start_date": dates.iloc[0].date().isoformat(), "evaluation_start": dates.iloc[calibration_days].date().isoformat(),
        "evaluation_end": dates.iloc[-1].date().isoformat(), "process_variance": process_variance,
        "observation_variance": observation_variance, "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="UCI Bike Sharing ZIP containing day.csv")
    parser.add_argument("--output", type=Path, required=True, help="New JSON result file")
    parser.add_argument("--particles", type=int, default=256)
    args = parser.parse_args()
    result = check_bike_sample(args.archive, args.particles)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
