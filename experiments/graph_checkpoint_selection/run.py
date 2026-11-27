"""Compare a validation-selected GCN checkpoint with its fixed final epoch."""

import argparse
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from diy_automl._graph_network import LEARNING_RATE, WEIGHT_DECAY, graph_forward, train_graph_layers
from diy_automl.graph_classification import gcn_loss, normalize_graph, read_graph, split_nodes
from diy_automl.reporting import fail, file_sha256, prepare, save, summary
from diy_automl.supervised import metrics


SCHEMA = "graph-checkpoint-selection/v1"
DEFAULT_SEEDS = (42, 43, 44, 45, 46)


def fit_checkpoints(features, adjacency, train, train_labels, validation, validation_labels, *, seed, epochs, hidden_size):
    classes = np.unique(train_labels)
    last_logits = None

    def validation_f1(logits):
        nonlocal last_logits
        last_logits = logits
        return metrics(validation_labels, logits[validation].argmax(axis=1), classes)["F1"]

    objective = partial(gcn_loss, train=train, train_labels=train_labels)
    weights, epoch, history = train_graph_layers(
        features, adjacency, output_size=len(classes), seed=seed, epochs=epochs, hidden_size=hidden_size,
        objective=objective, validation_score=validation_f1,
    )
    selected_logits, _, _ = graph_forward(adjacency @ features, adjacency, weights)
    return epoch, np.asarray(history), selected_logits.argmax(axis=1), last_logits.argmax(axis=1)


def compare_checkpoints(nodes, edges, output, *, split_seeds=DEFAULT_SEEDS, training_seeds=DEFAULT_SEEDS,
                        epochs=120, hidden_size=16, train_per_class=20, validation_fraction=0.2):
    split_seeds, training_seeds = tuple(split_seeds), tuple(training_seeds)
    for seeds in (split_seeds, training_seeds):
        if not seeds or min(seeds) < 0 or len(seeds) != len(set(seeds)):
            raise ValueError("Provide nonempty, unique, nonnegative seeds for each axis")
    if min(epochs, hidden_size, train_per_class) < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Use positive epoch, width and training counts, and a validation fraction between 0 and 1")

    inputs = {"nodes": str(nodes), "edges": str(edges)}
    hashes = {name: file_sha256(path) for name, path in inputs.items()}
    node_ids, class_names, labels, features, graph, _ = read_graph(nodes, edges)
    if any(file_sha256(path) != hashes[name] for name, path in inputs.items()):
        raise ValueError("Graph inputs changed while being read")
    if len(class_names) < 2:
        raise ValueError("Graph classification needs at least two classes")
    parts = [split_nodes(labels, train_per_class, validation_fraction, seed) for seed in split_seeds]
    settings = dict(
        inputs, split_seeds=split_seeds, training_seeds=training_seeds, epochs=epochs, hidden_size=hidden_size,
        train_per_class=train_per_class, validation_fraction=validation_fraction, cpu_threads=1,
        learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    output, manifest = prepare(output, SCHEMA, settings)
    try:
        adjacency = normalize_graph(graph)
        classes = np.arange(len(class_names))
        scores, records, selected_predictions, final_predictions = [], [], [], []
        with threadpool_limits(limits=1):
            for split_seed, partition in zip(split_seeds, parts, strict=True):
                train, validation, holdout = (partition[name] for name in ("train", "validation", "test"))
                for training_seed in training_seeds:
                    started = perf_counter()
                    selected, history, selected_pred, final_pred = fit_checkpoints(
                        features, adjacency, train, labels[train], validation, labels[validation],
                        seed=training_seed, epochs=epochs, hidden_size=hidden_size,
                    )
                    fit_seconds = perf_counter() - started
                    selected_f1 = metrics(labels[holdout], selected_pred[holdout], classes)["F1"]
                    final_f1 = metrics(labels[holdout], final_pred[holdout], classes)["F1"]
                    scores.append({
                        "Split seed": split_seed, "Training seed": training_seed,
                        "Selected epoch": selected, "Final epoch": epochs,
                        "Selected training loss": history[selected - 1, 1], "Final training loss": history[-1, 1],
                        "Selected validation F1": history[selected - 1, 2], "Final validation F1": history[-1, 2],
                        "Selected holdout F1": selected_f1, "Final holdout F1": final_f1,
                        "Holdout F1 difference": selected_f1 - final_f1, "Fit seconds": fit_seconds,
                    })
                    records.append(history)
                    selected_predictions.append(selected_pred)
                    final_predictions.append(final_pred)

        comparison = pd.DataFrame(scores)
        sheets = {
            "Checkpoint_Comparison": comparison,
            "Summary": summary([
                ("Protocol", "Transductive classification on one graph"),
                ("Selection", "Highest validation macro F1; first epoch on ties"),
                ("Reference", f"Fixed epoch {epochs} in the same run"),
                ("Paired runs", len(scores)),
                ("Mean holdout F1 difference", comparison["Holdout F1 difference"].mean()),
            ]),
        }
        arrays = {
            "node_ids": node_ids, "labels": labels,
            "split_seeds": np.repeat(split_seeds, len(training_seeds)),
            "training_seeds": np.tile(training_seeds, len(split_seeds)),
            **{name: np.stack([partition[name] for partition in parts]) for name in ("train", "validation", "test")},
            "training_history": np.stack(records),
            "selected_predictions": np.stack(selected_predictions), "final_predictions": np.stack(final_predictions),
        }
        manifest.update(
            input_sha256=hashes, classes=class_names,
            experiment_sha256=file_sha256(Path(__file__)),
        )
        return save(output, manifest, sheets, arrays)
    except Exception as error:
        fail(output, manifest, error)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=Path, required=True)
    parser.add_argument("--edges", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--training-seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--epochs", type=int, default=120)
    print(compare_checkpoints(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
