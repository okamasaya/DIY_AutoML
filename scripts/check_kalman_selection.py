"""Check the Kalman pilot decision and seasonal forecast candidates on public series."""

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from diy_automl import kalman_track_csv
from diy_automl.kalman import filter_series


AIR_SOURCE = "https://svn.r-project.org/R/trunk/src/library/datasets/data/AirPassengers.R"
TAXI_SOURCE = "https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/nyc_taxi.csv"
MIN_RMSE_GAIN = 0.05


def air_passengers(raw):
    match = re.fullmatch(
        r'\s*"AirPassengers"\s*<-\s*stats::ts\(\s*c\(([\d,\s]+)\)\s*,\s*1949\s*,\s*1960\.\d+\s*,\s*12\s*\)\s*',
        raw.decode("utf-8"),
    )
    if match is None:
        raise ValueError("Expected the R datasets AirPassengers source file")
    values = np.array([int(value) for value in match[1].split(",")], dtype=float)
    if len(values) != 144 or not (values > 0).all():
        raise ValueError("Expected 144 positive passenger counts")
    return pd.period_range("1949-01", periods=144, freq="M").astype(str).to_numpy(), values


def nyc_taxi(raw):
    data = pd.read_csv(BytesIO(raw), usecols=["timestamp", "value"])
    expected = pd.date_range("2014-07-01", "2015-01-31 23:30", freq="30min")
    values = data["value"].to_numpy(dtype=float)
    if not pd.DatetimeIndex(pd.to_datetime(data["timestamp"])).equals(expected) or not np.isfinite(values).all():
        raise ValueError("Expected the complete NAB half-hour NYC taxi series")
    return expected.strftime("%Y-%m-%d %H:%M").to_numpy(), values


def rmse(errors):
    return float(np.sqrt(np.mean(errors**2)))


def compare(output, name, path, source, loader, calibration, pilot_end, season, *, monthly=False):
    raw = path.read_bytes()
    dates, values = loader(raw)
    historical = values[:calibration]
    change_scale = float(np.mean(np.diff(historical)**2))
    options = {"observation_sd": float(np.sqrt(0.45 * change_scale)), "process_noise": 0.1 * change_scale,
               "initial_rate_sd": float(np.sqrt(change_scale))}

    # Start with the last calibration observation so the first pilot row can be scored.
    time = np.arange(calibration - 1, len(values))
    observed = values[time]
    pilot_rows = pilot_end - calibration + 1
    csv = output / f"{name}-input.csv"
    pd.DataFrame({"step": time, "timestamp": dates[time], "value": observed}).to_csv(csv, index=False)
    native = kalman_track_csv(
        csv, output / f"{name}-kalman-track", time_column="step", observation_column="value",
        pilot_rows=pilot_rows, min_rmse_gain=MIN_RMSE_GAIN, compact_report=True, **options,
    )
    decision = pd.read_excel(native / "report.xlsx", sheet_name="Decision").iloc[0]
    counterfactual = filter_series(time, observed, np.zeros_like(observed), **options)
    forecasts = {
        "Last observation": counterfactual["persistence"][1:],
        "Kalman": counterfactual["forecast"][1:, 0],
    }
    seasonal = values[calibration - season:-season]
    if monthly:
        forecasts["Same month last year"] = seasonal
        forecasts["Same month plus growth"] = seasonal + np.mean(historical[season:] - historical[:-season])
        forecasts["Same month times growth factor"] = seasonal * np.exp(np.mean(np.log(historical[season:] / historical[:-season])))
    else:
        forecasts["Same time last week"] = seasonal
        forecasts["Same time yesterday"] = values[calibration - 48:-48]

    split = pilot_end - calibration
    targets = values[calibration:]
    metrics = {method: {"pilot_rmse": rmse((forecast - targets)[:split]), "holdout_rmse": rmse((forecast - targets)[split:])}
               for method, forecast in forecasts.items()}
    selected = min(metrics, key=lambda method: metrics[method]["pilot_rmse"])
    points = pd.DataFrame({"timestamp": dates[calibration:], "period": np.where(np.arange(len(targets)) < split, "pilot", "holdout"),
                           "actual": targets, **forecasts})
    points.to_csv(output / f"{name}-forecasts.csv", index=False)
    return {
        "name": name,
        "unit": "thousands of passengers" if monthly else "passengers",
        "source": source,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "calibration": {"start": dates[0], "end": dates[calibration - 1], "rows": calibration},
        "pilot": {"start": dates[calibration], "end": dates[pilot_end - 1], "rows": split},
        "holdout": {"start": dates[pilot_end], "end": dates[-1], "rows": len(values) - pilot_end},
        "kalman_settings": options,
        "minimum_native_rmse_gain": MIN_RMSE_GAIN,
        "native_decision": {"selected": decision["Selected method"], "reason": decision["Reason"],
                            "paired_pilot_rows": int(decision["Paired observations"]),
                            "kalman_forecast_rows": int(decision["Kalman forecast steps"])},
        "selected_with_seasonal_candidates": selected,
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--air-passengers", type=Path, required=True)
    parser.add_argument("--nyc-taxi", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory for forecasts, reports and summary")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = [
        compare(args.output, "air-passengers", args.air_passengers, AIR_SOURCE, air_passengers, 60, 96, 12, monthly=True),
        compare(args.output, "nyc-taxi", args.nyc_taxi, TAXI_SOURCE, nyc_taxi, 8 * 336, 12 * 336, 336),
    ]
    with (args.output / "result.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
