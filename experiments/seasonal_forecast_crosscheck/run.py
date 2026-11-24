"""Compare one-step forecasts on the two published Kalman decision examples."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.voyager_crosscheck.run import autogluon, h2o_automl, write_json


SAMPLES = {"air-passengers": "AirPassengers.R", "nyc-taxi": "nyc_taxi.csv"}
DIY_MODES = {"DIY AutoML": "native", "DIY + seasonal candidates": "seasonal", "Kalman continued": "kalman"}
ENGINES = {"autogluon": "AutoGluon Tabular", "h2o": "H2O AutoML"}
KEY = ["dataset", "timestamp", "period"]


def feature_frame(name, dates, values):
    monthly = name == "air-passengers"
    frame = pd.DataFrame({"timestamp": dates, "actual": values, "step": np.arange(len(values))})
    lags = (1, 2, 3, 12) if monthly else (1, 2, 3, 48, 336)
    for lag in lags:
        frame[f"lag_{lag}"] = frame["actual"].shift(lag)

    # The target timestamp is on a fixed schedule and is known at the forecast origin.
    time = pd.to_datetime(dates, format="%Y-%m" if monthly else "%Y-%m-%d %H:%M")
    cycles = {"month": (time.month - 1, 12)} if monthly else {
        "day": (time.hour * 2 + time.minute // 30, 48), "week": (time.dayofweek, 7),
    }
    for label, (position, length) in cycles.items():
        angle = 2 * np.pi * position / length
        frame[f"{label}_sin"] = np.sin(angle)
        frame[f"{label}_cos"] = np.cos(angle)
    return frame


def prepare(source, reference, run):
    from scripts.check_kalman_selection import air_passengers, nyc_taxi

    prior = json.loads((reference / "result.json").read_text(encoding="utf-8"))
    loaders = {"air-passengers": air_passengers, "nyc-taxi": nyc_taxi}
    datasets, metadata = {}, {}
    for item in prior:
        name = item["name"]
        raw = (source / SAMPLES[name]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != item["source_sha256"]:
            raise ValueError(f"Source checksum differs from the DIY run for {name}")
        dates, values = loaders[name](raw)
        frame = feature_frame(name, dates, values)
        features = [column for column in frame if column not in ("timestamp", "actual")]
        counts = [item[period]["rows"] for period in ("calibration", "pilot", "holdout")]
        frame["period"] = np.repeat(["train", "pilot", "holdout"], counts)
        frame["dataset"] = name
        previous = pd.read_csv(reference / f"{name}-forecasts.csv", dtype={"timestamp": str})
        scored = frame[frame["period"] != "train"]
        if (not np.array_equal(scored["timestamp"], previous["timestamp"])
                or not np.array_equal(scored["period"], previous["period"])
                or not np.array_equal(scored["actual"], previous["actual"])):
            raise ValueError(f"Split or observed values differ from the DIY run for {name}")
        selections = {"native": item["native_decision"]["selected"],
                      "seasonal": item["selected_with_seasonal_candidates"], "kalman": "Kalman"}
        for mode, candidate in selections.items():
            frame.loc[scored.index, mode] = previous[candidate].to_numpy()
        datasets[name] = frame
        metadata[name] = {"features": features, "unit": item["unit"], "source_sha256": item["source_sha256"],
                          "selections": selections,
                          "reference_forecasts_sha256": hashlib.sha256((reference / f"{name}-forecasts.csv").read_bytes()).hexdigest()}
    if datasets.keys() != SAMPLES.keys():
        raise ValueError("Expected both public series in the DIY reference run")
    run.mkdir(parents=True, exist_ok=False)
    for name, frame in datasets.items():
        frame.to_csv(run / f"{name}.csv", index=False)
    write_json(run / "input.json", metadata)


def predict(engine, run, metadata):
    all_rows, all_details = [], {}
    for name in SAMPLES:
        data = pd.read_csv(run / f"{name}.csv", dtype={"timestamp": str})
        train = data[data["period"] == "train"]
        pilot = data[data["period"] == "pilot"]
        score = data[data["period"] != "train"]
        variables = dict(features=metadata[name]["features"], targets=["actual"])
        if engine == "autogluon":
            estimates, details = autogluon(run / name, train, pilot, score, **variables)
        else:
            estimates, details = h2o_automl(train, pilot, score, **variables)
        forecast = estimates[ENGINES[engine]]
        if forecast.shape != (len(score), 1) or not np.isfinite(forecast).all():
            raise ValueError(f"Invalid forecasts from {engine} on {name}")
        all_rows.append(score[KEY].assign(model=ENGINES[engine], forecast=forecast[:, 0]))
        all_details[name] = details
    pd.concat(all_rows).to_csv(run / f"{engine}.csv", index=False, mode="x")
    write_json(run / f"{engine}.json", all_details)


def report(run, metadata):
    series = pd.concat([pd.read_csv(run / f"{name}.csv", dtype={"timestamp": str}) for name in SAMPLES], ignore_index=True)
    score = series[series["period"] != "train"]
    points = [score[KEY].assign(model=model, forecast=score[mode]) for model, mode in DIY_MODES.items()]
    points.extend(pd.read_csv(run / f"{engine}.csv", dtype={"timestamp": str}) for engine in ENGINES)
    predictions = pd.concat(points, ignore_index=True)
    if set(predictions["model"]) != {*DIY_MODES, *ENGINES.values()} or not np.isfinite(predictions["forecast"]).all():
        raise ValueError("Expected finite forecasts from all comparison models")
    expected = pd.MultiIndex.from_frame(score[KEY])
    for model, comparison in predictions.groupby("model"):
        keys = pd.MultiIndex.from_frame(comparison[KEY])
        if keys.has_duplicates or len(keys) != len(expected) or set(keys) != set(expected):
            raise ValueError(f"Prediction timestamps differ for {model}")
    predictions = predictions.merge(score[KEY + ["actual"]], on=KEY, validate="many_to_one")
    predictions["absolute_error"] = np.abs(predictions["forecast"] - predictions["actual"])

    results = []
    for (dataset, model), rows in predictions.groupby(["dataset", "model"], sort=False):
        pilot = rows.loc[rows["period"] == "pilot", "absolute_error"]
        holdout = rows.loc[rows["period"] == "holdout", "absolute_error"]
        p90 = float(pilot.quantile(0.90))
        results.append({
            "dataset": dataset, "unit": metadata[dataset]["unit"], "model": model,
            "pilot_rows": len(pilot), "holdout_rows": len(holdout),
            "pilot_rmse": float(np.sqrt(np.mean(pilot**2))), "holdout_rmse": float(np.sqrt(np.mean(holdout**2))),
            "holdout_mae": float(holdout.mean()), "pilot_absolute_error_p90": p90,
            "holdout_share_within_pilot_p90": float(np.mean(holdout <= p90)),
        })
    predictions.to_csv(run / "predictions.csv", index=False, mode="x")
    pd.DataFrame(results).to_csv(run / "metrics.csv", index=False, mode="x")
    write_json(run / "comparison.json", {
        "input": metadata,
        "engines": {engine: json.loads((run / f"{engine}.json").read_text(encoding="utf-8")) for engine in ENGINES},
        "metrics": results,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", *ENGINES, "report"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path("runs/seasonal-selection-20261124"))
    parser.add_argument("--reference", type=Path, default=Path("runs/seasonal-selection-20261124/evaluation"))
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.source, args.reference, args.run)
    else:
        metadata = json.loads((args.run / "input.json").read_text(encoding="utf-8"))
        if args.action == "report":
            report(args.run, metadata)
        else:
            predict(args.action, args.run, metadata)
    print(args.run)


if __name__ == "__main__":
    main()
