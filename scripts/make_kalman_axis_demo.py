"""Generate a reproducible, simulated one-axis trajectory with sensor gaps."""

import argparse
import csv
from pathlib import Path

import numpy as np


def make_trajectory(seed):
    rng = np.random.default_rng(seed)
    times = np.arange(601, dtype=float) * 0.1
    acceleration = 0.35 * np.sin(times * 0.65) + 0.2 * np.sin(times * 1.1)
    angle, rate = 0.0, 0.15
    for index, (time, command) in enumerate(zip(times, acceleration)):
        reading = "" if index % 31 in (10, 11, 12) else angle + rng.normal(0.0, 0.1)
        yield time, reading, angle, command
        # A small simulated disturbance is unknown to the open-loop predictor.
        actual = command + rng.normal(0.0, 0.2)
        angle += 0.1 * rate + 0.005 * actual
        rate += 0.1 * actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s", "observed_angle", "true_angle", "commanded_acceleration"])
        writer.writerows(make_trajectory(args.seed))


if __name__ == "__main__":
    main()
