"""Reproduce one-step forecasts on a small UCI Bike Sharing daily sample."""

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from diy_automl.local_level_check import compare_local_level


def check_bike_sample(archive, particles):
    with ZipFile(archive) as source:
        data = source.read("day.csv")
    daily = pd.read_csv(BytesIO(data), usecols=["dteday", "cnt"], parse_dates=["dteday"])
    daily = daily.sort_values("dteday").head(150)
    dates = daily["dteday"]
    counts = daily["cnt"].to_numpy(dtype=float)
    if len(daily) != 150 or not dates.diff().iloc[1:].eq(pd.Timedelta(days=1)).all() or not np.isfinite(counts).all():
        raise ValueError("The sample needs 150 consecutive days with finite rental counts")

    calibration_days = 60
    comparison = compare_local_level(counts, calibration_days, particles)
    particle = comparison["particle_runs"][0].copy()
    particle_seed = particle.pop("seed")
    return {
        "day_csv_sha256": hashlib.sha256(data).hexdigest(), "particles": particles, "particle_seed": particle_seed,
        "calibration_days": calibration_days, "evaluation_days": len(counts) - calibration_days,
        "start_date": dates.iloc[0].date().isoformat(), "evaluation_start": dates.iloc[calibration_days].date().isoformat(),
        "evaluation_end": dates.iloc[-1].date().isoformat(), "process_variance": comparison["process_variance"],
        "observation_variance": comparison["observation_variance"], "metrics": [*comparison["metrics"], particle],
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
