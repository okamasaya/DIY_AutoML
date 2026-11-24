"""Run a chronological position forecast comparison using one synthetic Voyager arc."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd


SEED = 42
STEP_DAYS = 7
AU_KM = 149_597_870.7
SENSORS = ["range_au", "range_rate_au_per_year"]
OBSERVER = ["earth_x_au", "earth_y_au", "earth_vx_au_per_year", "earth_vy_au_per_year"]
FEATURES = ["origin_day", *OBSERVER, *SENSORS, *(f"previous_{x}" for x in SENSORS)]
TARGETS = ["x_au", "y_au"]
KEY = ["scenario", "origin_day", "target_day", "split"]
ENGINES = ("diy", "autogluon", "h2o")


def prepare(source, run):
    truth_path, observations_path = source / "truth.csv", source / "observations.csv"
    truth = pd.read_csv(truth_path).set_index("time_day")
    observations = pd.read_csv(observations_path)
    nominal, bias = (observations[observations["scenario"] == name].sort_values("time_day").copy()
                     for name in ("nominal", "bias"))
    days = np.arange(7, 729, 7)
    if (not np.array_equal(truth.index, np.r_[0, days])
            or any(not np.array_equal(frame["time_day"], days) for frame in (nominal, bias))
            or not np.isfinite(truth[TARGETS].to_numpy()).all()
            or not np.isfinite(pd.concat([nominal, bias])[OBSERVER + SENSORS].to_numpy()).all()):
        raise ValueError("Expected the complete Voyager synthetic sample on its seven-day schedule")
    replay = nominal.assign(scenario="replayed")
    older = nominal.loc[nominal["time_day"] + STEP_DAYS <= 350, SENSORS].to_numpy()
    damaged = replay["time_day"] >= 420
    indices = np.random.default_rng(SEED).integers(len(older), size=int(damaged.sum()))
    replay.loc[damaged, SENSORS] = older[indices]

    frames = []
    for data in (nominal, bias, replay):
        data = data.rename(columns={"time_day": "origin_day"}).reset_index(drop=True)
        data[[f"previous_{x}" for x in SENSORS]] = data[SENSORS].shift()
        data["target_day"] = data["origin_day"] + STEP_DAYS
        data = data.join(truth[TARGETS], on="target_day")
        data["split"] = pd.cut(data["target_day"], [0, 350, 420, 728, 735],
                               labels=["train", "tune", "test", "end"])
        frames.append(data)
    run.mkdir(parents=True, exist_ok=False)
    pd.concat(frames, ignore_index=True).to_csv(run / "dataset.csv", index=False)
    write_json(run / "input.json", {
        "source": str(source), "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                                 for p in (truth_path, observations_path)},
        "train_end_day": 350, "tune_end_day": 420, "test_end_day": 728,
        "replay_seed": SEED, "replay_start_origin_day": 420, "replay_source_origin_days": days[indices].tolist(),
        "features": FEATURES, "targets": TARGETS,
    })


def splits(data):
    nominal = data["scenario"] == "nominal"
    available = data[FEATURES].notna().all(axis=1)
    train = data[nominal & available & data["split"].eq("train")]
    tune = data[nominal & data["split"].eq("tune")]
    score = data[data["split"].eq("test") | (nominal & data["split"].eq("tune"))]
    return train, tune, score


def diy(data, train, score):
    # Reuse the physical model and prior from the Voyager example.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from make_voyager_demo import (OBSERVATION_STD, PRIOR_MEAN, PRIOR_STD, PROCESS_STD,
                                   measurement, measurement_jacobian, transition, transition_jacobian)
    from diy_automl import __version__, extended_kalman_filter

    forecasts = pd.DataFrame(index=data.index, columns=TARGETS, dtype=float)
    for _, scenario in data.groupby("scenario", sort=False):
        earth = scenario[OBSERVER].to_numpy()
        result = extended_kalman_filter(
            scenario[SENSORS].to_numpy(), initial_mean=PRIOR_MEAN, initial_covariance=np.diag(PRIOR_STD**2),
            transition=transition, transition_jacobian=transition_jacobian,
            measurement=lambda x, k: measurement(x, earth[k - 1]),
            measurement_jacobian=lambda x, k: measurement_jacobian(x, earth[k - 1]),
            process_covariance=np.diag(PROCESS_STD**2), observation_covariance=np.diag(OBSERVATION_STD**2),
        )
        # Row i's forecast is stored at step i+1, before that step's observations are incorporated.
        forecasts.loc[scenario.index[:-1], TARGETS] = result["forecast"][1:, :2]
    state = PRIOR_MEAN
    orbit = {}
    for step, day in enumerate(range(7, 729, 7), 1):
        state = transition(state, step)
        orbit[day] = state[:2]
    open_loop = np.stack([orbit[day] for day in score["target_day"]])
    trend = np.column_stack([np.polyval(np.polyfit(train["target_day"], train[name], 1), score["target_day"])
                            for name in TARGETS])
    return {"DIY EKF": forecasts.loc[score.index].to_numpy(), "Physics only": open_loop, "Time trend": trend}, {
        "version": __version__, "prior_state": PRIOR_MEAN.tolist(),
    }


def autogluon(run, train, tune, score):
    from importlib.metadata import version
    from autogluon.tabular import TabularPredictor

    forecasts, selected = [], {}
    for target in TARGETS:
        predictor = TabularPredictor(label=target, problem_type="regression", eval_metric="root_mean_squared_error",
                                     path=str(run / "ag_models" / target), verbosity=1)
        predictor.fit(train[FEATURES + [target]], tuning_data=tune[FEATURES + [target]], num_cpus=1, num_gpus=0,
                      num_bag_folds=0, num_stack_levels=0,
                      hyperparameters={"LR": {}, "RF": {"n_estimators": 64}, "XT": {"n_estimators": 64}})
        selected[target] = predictor.model_best
        forecasts.append(predictor.predict(score[FEATURES]).to_numpy())
    return {"AutoGluon Tabular": np.column_stack(forecasts)}, {"version": version("autogluon.tabular"), "selected": selected}


def h2o_automl(train, tune, score):
    import h2o
    from h2o.automl import H2OAutoML
    from h2o.backend import H2OLocalServer

    # H2O parses this argument as a URI. Use a relative path without reserved characters.
    with TemporaryDirectory(prefix=".voyager-h2o-", dir=".") as ice_root:
        server = H2OLocalServer.start(
            nthreads=1, max_mem_size=1024**3, ice_root=Path(ice_root).name, bind_to_localhost=True, verbose=False,
            jvm_custom_args=["-Dfile.encoding=UTF-8", "-Dsun.stdout.encoding=UTF-8", "-Dsun.stderr.encoding=UTF-8",
                             "-Dsys.ai.h2o.telemetry.disabled=true"],
        )
        try:
            h2o.connect(server=server, verbose=False, telemetry=False)
            h2o.no_progress()
            forecasts, selected = [], {}
            features_only = h2o.H2OFrame(score[FEATURES])
            for target in TARGETS:
                training = h2o.H2OFrame(train[FEATURES + [target]])
                validation = h2o.H2OFrame(tune[FEATURES + [target]])
                automl = H2OAutoML(include_algos=["GLM", "DRF", "GBM"], max_models=4, nfolds=0,
                                   sort_metric="RMSE", seed=SEED)
                automl.train(x=FEATURES, y=target, training_frame=training,
                             validation_frame=validation, leaderboard_frame=validation)
                leader = automl.leader
                selected[target] = leader.algo
                forecasts.append(leader.predict(features_only).as_data_frame()["predict"].to_numpy())
            return {"H2O AutoML": np.column_stack(forecasts)}, {"version": h2o.__version__, "selected": selected}
        finally:
            server.shutdown()


def run_engine(engine, run, data):
    train, tune, score = splits(data)
    if engine == "diy":
        estimates, details = diy(data, train, score)
    elif engine == "autogluon":
        estimates, details = autogluon(run, train, tune, score)
    else:
        estimates, details = h2o_automl(train, tune, score)
    rows = []
    for name, forecast in estimates.items():
        if forecast.shape != (len(score), 2) or not np.isfinite(forecast).all():
            raise ValueError(f"Invalid position forecasts from {name}")
        frame = score[KEY].assign(model=name)
        frame[["pred_x_au", "pred_y_au"]] = forecast
        rows.append(frame)
    pd.concat(rows).to_csv(run / f"{engine}.csv", index=False, mode="x")
    write_json(run / f"{engine}.json", details)


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def report(run, data):
    train, tune, score = splits(data)
    predictions = pd.concat([pd.read_csv(run / f"{engine}.csv") for engine in ENGINES], ignore_index=True)
    expected = pd.MultiIndex.from_frame(score[KEY])
    for model, values in predictions.groupby("model"):
        keys = pd.MultiIndex.from_frame(values[KEY])
        if len(keys) != len(expected) or keys.has_duplicates or set(keys) != set(expected):
            raise ValueError(f"Forecast timestamps differ for {model}")
    predictions = predictions.merge(score[KEY + TARGETS], on=KEY, validate="many_to_one")
    delta = predictions[["pred_x_au", "pred_y_au"]].to_numpy() - predictions[TARGETS].to_numpy()
    predictions["position_error_km"] = AU_KM * np.linalg.norm(delta, axis=1)

    metrics = []
    for name, model in predictions.groupby("model", sort=False):
        errors = model.loc[model["split"] == "tune", "position_error_km"]
        radius = float(errors.quantile(0.90))
        for scenario, rows in model[model["split"] == "test"].groupby("scenario", sort=False):
            values = rows["position_error_km"]
            metrics.append({"model": name, "scenario": scenario, "test_rows": len(rows),
                            "position_rmse_km": float(np.sqrt(np.mean(values**2))),
                            "tune_position_rmse_km": float(np.sqrt(np.mean(errors**2))),
                            "tune_error_p90_km": radius, "test_share_within_tune_p90": float(np.mean(values <= radius))})

    # Range increments should approximately match the integrated range rate.
    step_years = STEP_DAYS / 365.25
    mismatch = np.abs((data[SENSORS[0]] - data[f"previous_{SENSORS[0]}"])
                      - step_years / 2 * (data[SENSORS[1]] + data[f"previous_{SENSORS[1]}"]))
    threshold = float(mismatch.loc[tune.index].quantile(0.95))
    quality = []
    for scenario, rows in score[score["split"] == "test"].groupby("scenario", sort=False):
        count = int((mismatch.loc[rows.index] > threshold).sum())
        quality.append({"scenario": scenario, "test_rows": len(rows), "flagged_rows": count,
                        "range_rate_consistency_threshold_au": threshold})

    predictions.to_csv(run / "predictions.csv", index=False, mode="x")
    pd.DataFrame(metrics).to_csv(run / "metrics.csv", index=False, mode="x")
    pd.DataFrame(quality).to_csv(run / "quality.csv", index=False, mode="x")
    write_json(run / "comparison.json", {"train_rows": len(train), "tune_rows": len(tune),
               "engines": {engine: json.loads((run / f"{engine}.json").read_text(encoding="utf-8")) for engine in ENGINES},
               "metrics": metrics, "quality": quality})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", *ENGINES, "report"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path("runs/voyager-synthetic-20261124"))
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.source, args.run)
    else:
        data = pd.read_csv(args.run / "dataset.csv")
        if args.action == "report":
            report(args.run, data)
        else:
            run_engine(args.action, args.run, data)
    print(args.run)


if __name__ == "__main__":
    main()
