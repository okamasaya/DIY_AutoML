"""Compare one-month-ahead forecasts with shared trend and monthly seasonality."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from check_public_particle_samples import CALIBRATION_ROWS, CO2_SOURCE, PARTICLE_SEEDS, mauna_loa_sample
from diy_automl.local_level_check import compare_local_level


def fit_seasonal_trend(values, calibration_rows):
    """Fit one slope and monthly offsets relative to January on a January-start series."""
    elapsed = np.arange(len(values))
    monthly_indicators = elapsed[:, None] % 12 == np.arange(1, 12)
    design = np.column_stack((np.ones(len(values)), elapsed, monthly_indicators))
    coefficients, _, _, _ = np.linalg.lstsq(design[:calibration_rows], values[:calibration_rows], rcond=None)
    return design @ coefficients, {
        "initial_january_ppm": float(coefficients[0]),
        "trend_ppm_per_month": float(coefficients[1]),
        "monthly_offsets_ppm": [0.0, *coefficients[2:].tolist()],
    }


def compare_window(raw, start_year, particles):
    dates, values = mauna_loa_sample(raw, start_year=start_year)
    unadjusted = compare_local_level(values, CALIBRATION_ROWS, particles, particle_seeds=PARTICLE_SEEDS, season=12)
    fixed_forecast, coefficients = fit_seasonal_trend(values, CALIBRATION_ROWS)
    residuals = values - fixed_forecast
    # Adding the same fixed forecast to the residual observations, predictions and
    # bounds preserves every error, coverage indicator and interval width.
    adjusted = compare_local_level(residuals, CALIBRATION_ROWS, particles, particle_seeds=PARTICLE_SEEDS)
    adjusted["metrics"][0]["method"] = "Last residual"
    return {
        "calibration_start": dates[0],
        "calibration_end": dates[CALIBRATION_ROWS - 1],
        "evaluation_start": dates[CALIBRATION_ROWS],
        "evaluation_end": dates[-1],
        "evaluation_rows": len(values) - CALIBRATION_ROWS,
        "seasonal_trend_coefficients": coefficients,
        "fixed_seasonal_trend_rmse": float(np.sqrt(np.mean(residuals[CALIBRATION_ROWS:] ** 2))),
        "unadjusted": unadjusted,
        "seasonal_trend": adjusted,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--co2", type=Path, required=True, help="NOAA co2_mm_mlo.csv")
    parser.add_argument("--particles", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True, help="New JSON result file")
    args = parser.parse_args()
    raw = args.co2.read_bytes()
    result = {
        "series": "Mauna Loa CO2 monthly average",
        "unit": "ppm",
        "source": CO2_SOURCE,
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "calibration_rows": CALIBRATION_ROWS,
        "particles": args.particles,
        "windows": [compare_window(raw, start_year, args.particles) for start_year in (1980, 1990)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
