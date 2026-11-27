"""Compare graph edges with otherwise identical models on four small datasets."""

import argparse
from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from diy_automl._graph_network import WEIGHT_DECAY, graph_forward, graph_gradients, train_graph_layers
from diy_automl.graph_classification import fit_gcn, fit_logistic, split_nodes
from diy_automl.reporting import file_sha256, write_json
from diy_automl.supervised import metrics


SEEDS = (42, 43, 44)
EPOCHS = 120
HIDDEN_SIZE = 16
AU_KM = 149_597_870.7


@dataclass(frozen=True)
class ForecastFrame:
    features: np.ndarray
    targets: np.ndarray
    periods: np.ndarray
    keys: pd.DataFrame


def normalize_rows(graph):
    degrees = np.asarray(graph.sum(axis=1)).ravel()
    return (sparse.diags(1 / degrees) @ graph).tocsr()


def causal_graph(size, lags):
    offsets = [0, *(-lag for lag in lags if lag < size)]
    return normalize_rows(sparse.diags([1.0] * len(offsets), offsets, shape=(size, size), format="csr"))


def position_rmse(residual):
    return float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))


def regression_loss(messages, adjacency, weights, train, target):
    prediction, hidden, neighborhood = graph_forward(messages, adjacency, weights)
    error = prediction[train] - target
    loss = 0.5 * np.mean(error**2) + WEIGHT_DECAY / 2 * np.sum(weights[0]**2)
    derivative = np.zeros_like(prediction)
    derivative[train] = error / error.size
    return loss, graph_gradients(messages, adjacency, weights, hidden, neighborhood, derivative), prediction


def compare_regression(name, frames, graph, output, *, unit, scale=1.0):
    nominal = frames["nominal"]
    train = np.flatnonzero(nominal.periods == "train")
    validation = np.flatnonzero(nominal.periods == "validation")
    feature_scaler = StandardScaler().fit(nominal.features[train])
    target_scaler = StandardScaler().fit(nominal.targets[train])
    inputs = {key: np.column_stack((np.nan_to_num(feature_scaler.transform(frame.features), nan=0),
                                    np.ones(len(frame.features)))) for key, frame in frames.items()}
    target = target_scaler.transform(nominal.targets)
    test_results, traces = [], []
    for method, adjacency in (("No-edge network", sparse.eye(graph.shape[0], format="csr")), ("GCN", graph)):
        for seed in SEEDS:
            weights, epoch, _ = train_graph_layers(
                inputs["nominal"], adjacency, output_size=target.shape[1], seed=seed, epochs=EPOCHS, hidden_size=HIDDEN_SIZE,
                objective=lambda x, a, w: regression_loss(x, a, w, train, target[train]),
                validation_score=lambda z: -position_rmse((z[validation] - target[validation]) * target_scaler.scale_),
            )
            for scenario, frame in frames.items():
                prediction = target_scaler.inverse_transform(graph_forward(adjacency @ inputs[scenario], adjacency, weights)[0])
                test = frame.periods == "test"
                error = (prediction[test] - frame.targets[test]) * scale
                test_results.append({"dataset": name, "unit": unit, "method": method, "scenario": scenario,
                                     "seed": seed, "checkpoint": epoch, "test_rows": int(test.sum()),
                                     "test_rmse": position_rmse(error)})
                trace = frame.keys.loc[test].assign(method=method, scenario=scenario, seed=seed)
                for index in range(frame.targets.shape[1]):
                    trace[f"actual_{index}"] = frame.targets[test, index]
                    trace[f"predicted_{index}"] = prediction[test, index]
                traces.append(trace)
    pd.concat(traces, ignore_index=True).to_csv(output / f"{name}-predictions.csv", index=False)
    return test_results


def voyager_frames(path):
    from experiments.voyager_crosscheck.run import FEATURES, TARGETS

    data = pd.read_csv(path)
    frames = {}
    for scenario, rows in data.groupby("scenario", sort=False):
        rows = rows.sort_values("origin_day").reset_index(drop=True)
        stage = rows["split"].replace({"tune": "validation"}).to_numpy(copy=True)
        stage[(stage == "train") & rows[FEATURES].isna().any(axis=1)] = "warmup"
        frames[scenario] = ForecastFrame(rows[FEATURES].to_numpy(), rows[TARGETS].to_numpy(), stage,
                                         rows[["origin_day", "target_day"]])
    return frames, causal_graph(len(frames["nominal"].periods), (1, 2, 3))


def taxi_frames(path, metadata):
    data = pd.read_csv(path)
    features = metadata["nyc-taxi"]["features"]
    periods = data["period"].replace({"pilot": "validation", "holdout": "test"}).to_numpy()
    frame = ForecastFrame(data[features].to_numpy(), data[["actual"]].to_numpy(), periods, data[["timestamp"]])
    return {"nominal": frame}, causal_graph(len(data), (1, 48, 336))


def pedal_frames(path):
    with ZipFile(path) as outer, ZipFile(BytesIO(outer.read("pedalme_.zip"))) as archive:
        counts = pd.read_csv(archive.open("pedalme_features.csv"))
        edges = pd.read_csv(archive.open("pedalme_edges.csv"))
    series = counts.pivot(index="time", columns="location", values="demand").sort_index().sort_index(axis=1)
    weights = edges.pivot(index="from", columns="to", values="weight").reindex(index=series.columns, columns=series.columns)
    if (not np.array_equal(series.index, np.arange(36)) or series.shape[1] != 15
            or not np.isfinite(series).all().all() or not np.isfinite(weights).all().all()
            or (weights < 0).any().any() or (weights.sum(axis=1) <= 0).any()):
        raise ValueError("Unexpected Pedal Me weekly data or adjacency weights")
    weeks = series.index[3:].to_numpy()
    regions = series.columns.to_numpy()
    demand = series.to_numpy()
    features = np.column_stack([demand[3 - lag:-lag].ravel() for lag in (1, 2, 3)]
                               + [np.tile(np.eye(len(regions)), (len(weeks), 1))])
    keys = pd.DataFrame({"target_week": np.repeat(weeks, len(regions)), "region": np.tile(regions, len(weeks))})
    periods = np.select([keys["target_week"] < 22, keys["target_week"] < 28], ["train", "validation"], default="test")
    frame = ForecastFrame(features, demand[3:].reshape(-1, 1), periods, keys)
    graph = sparse.kron(sparse.eye(len(weeks)), normalize_rows(sparse.csr_matrix(weights.to_numpy())), format="csr")
    holdout = periods == "test"
    persistence = position_rmse(frame.targets[holdout] - features[holdout, :1])
    return {"nominal": frame}, graph, persistence


def compare_iris(path, output):
    data = pd.read_csv(path)
    labels = data.pop("target").to_numpy(dtype=int)
    scores, traces = [], []
    for seed in SEEDS:
        parts = split_nodes(labels, 20, 0.2, seed)
        test = parts["test"]
        values = StandardScaler().fit(data.iloc[parts["train"]]).transform(data)
        features = sparse.csr_matrix(values)
        neighbors = NearestNeighbors(n_neighbors=5).fit(values).kneighbors_graph(mode="connectivity")
        graph = neighbors.maximum(neighbors.T)
        logistic, _, _ = fit_logistic(features, labels, parts)
        for model, prediction in (("Logistic regression", logistic),
                                  ("No-edge network", fit_gcn(features, sparse.csr_matrix(graph.shape), labels, parts,
                                                              seed=seed, epochs=EPOCHS, hidden_size=HIDDEN_SIZE)[0]),
                                  ("GCN", fit_gcn(features, graph, labels, parts, seed=seed, epochs=EPOCHS,
                                                  hidden_size=HIDDEN_SIZE)[0])):
            scores.append({"seed": seed, "method": model, "test_rows": len(test),
                           **metrics(labels[test], prediction[test], np.unique(labels))})
            traces.append(pd.DataFrame({"row": test, "seed": seed, "method": model, "actual": labels[test],
                                        "predicted": prediction[test]}))
    pd.concat(traces, ignore_index=True).to_csv(output / "iris-predictions.csv", index=False)
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voyager", type=Path, default=Path("runs/voyager-crosscheck-20261124"))
    parser.add_argument("--taxi", type=Path, default=Path("runs/seasonal-forecast-crosscheck-20261124"))
    parser.add_argument("--iris", type=Path, default=Path("data/reference/iris.csv"))
    parser.add_argument("--pedal-zip", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    voyager_path, taxi_path, taxi_input = args.voyager / "dataset.csv", args.taxi / "nyc-taxi.csv", args.taxi / "input.json"
    voyager, voyage_graph = voyager_frames(voyager_path)
    taxi, taxi_graph = taxi_frames(taxi_path, json.loads(taxi_input.read_text(encoding="utf-8")))
    pedal, pedal_graph, persistence = pedal_frames(args.pedal_zip)
    args.output.mkdir(parents=True, exist_ok=False)
    forecasts = compare_regression("voyager", voyager, voyage_graph, args.output, unit="km", scale=AU_KM)
    forecasts += compare_regression("nyc-taxi", taxi, taxi_graph, args.output, unit="passengers")
    forecasts += compare_regression("pedal-me", pedal, pedal_graph, args.output, unit="deliveries")
    write_json(args.output / "results.json", {
        "input_sha256": {str(path): file_sha256(path) for path in (voyager_path, taxi_path, taxi_input, args.iris, args.pedal_zip)},
        "seeds": SEEDS, "epochs": EPOCHS, "hidden_size": HIDDEN_SIZE, "regression": forecasts,
        "pedal_me_persistence_test_rmse": persistence, "iris": compare_iris(args.iris, args.output),
    })
    print(args.output)


if __name__ == "__main__":
    main()
