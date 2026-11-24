"""Compare causal forecasts on R's Nile and NOAA's Mauna Loa monthly records."""

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from diy_automl.local_level_check import compare_local_level


NILE_SOURCE = "https://svn.r-project.org/R/trunk/src/library/datasets/data/Nile.R"
CO2_SOURCE = "https://www.gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv"
CALIBRATION_ROWS = 60
PARTICLE_SEEDS = (42, 43, 44, 45, 46)


def nile_sample(data):
    # Parse the numeric series without running the bundled R expression.
    match = re.fullmatch(
        r"\s*Nile\s*<-\s*stats::ts\(\s*c\(([\d,\s]+)\)\s*,\s*start\s*=\s*1871\s*\)\s*",
        data.decode("utf-8"),
    )
    if match is None:
        raise ValueError("Expected the R datasets Nile source")
    values = np.array([float(value) for value in match[1].split(",")])
    if len(values) != 100:
        raise ValueError("Expected 100 annual Nile values")
    return [str(year) for year in range(1871, 1971)], values


def mauna_loa_sample(data, *, start_year=1980):
    monthly = pd.read_csv(
        BytesIO(data), comment="#", skipinitialspace=True, usecols=["year", "month", "average"],
    )
    end_year = start_year + 9
    monthly = monthly[monthly["year"].between(start_year, end_year)].sort_values(["year", "month"])
    dates = pd.to_datetime(monthly[["year", "month"]].assign(day=1)).dt.to_period("M")
    expected = pd.period_range(f"{start_year}-01", f"{end_year}-12", freq="M")
    values = monthly["average"].to_numpy(dtype=float)
    if not pd.PeriodIndex(dates).equals(expected) or not (np.isfinite(values) & (values > 0)).all():
        raise ValueError(f"Expected complete monthly CO2 averages for {start_year}–{end_year}")
    return dates.astype(str).tolist(), values


def check_samples(nile_path, co2_path, particles):
    records = []
    for name, path, source, loader, unit, season in (
        ("Nile", nile_path, NILE_SOURCE, nile_sample, "10^8 m^3", None),
        ("Mauna Loa CO2", co2_path, CO2_SOURCE, mauna_loa_sample, "ppm", 12),
    ):
        raw = path.read_bytes()
        dates, values = loader(raw)
        result = compare_local_level(
            values, CALIBRATION_ROWS, particles, particle_seeds=PARTICLE_SEEDS, season=season,
        )
        records.append({
            "series": name,
            "unit": unit,
            "source": source,
            "input_sha256": hashlib.sha256(raw).hexdigest(),
            "calibration_start": dates[0],
            "calibration_end": dates[CALIBRATION_ROWS - 1],
            "evaluation_start": dates[CALIBRATION_ROWS],
            "evaluation_end": dates[-1],
            "calibration_rows": CALIBRATION_ROWS,
            "evaluation_rows": len(dates) - CALIBRATION_ROWS,
            "particles": particles,
            **result,
        })
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nile", type=Path, required=True, help="R Project Nile.R source file")
    parser.add_argument("--co2", type=Path, required=True, help="NOAA co2_mm_mlo.csv")
    parser.add_argument("--particles", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True, help="New JSON result file")
    args = parser.parse_args()
    records = check_samples(args.nile, args.co2, args.particles)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(records, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
