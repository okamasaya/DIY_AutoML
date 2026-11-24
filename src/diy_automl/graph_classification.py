"""Transductive node classification on a Cora-style graph."""

import csv
from numbers import Integral
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression

from ._graph_network import LEARNING_RATE, WEIGHT_DECAY, graph_forward, graph_gradients, train_graph_layers
from .reporting import fail, file_sha256, prepare, save, summary
from .supervised import metrics


SCHEMA = 22
LOGISTIC_C = (1.0, 10.0, 100.0)
METHODS = {"logistic": "Logistic regression", "gcn": "GCN"}


def read_graph(nodes, edges):
    content = pd.read_csv(nodes, sep="\t", header=None, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if content.shape[1] < 3:
        raise ValueError("Node rows need an ID, numeric features and a class label")
    node_ids = content.iloc[:, 0].to_numpy(dtype=str)
    labels = content.iloc[:, -1].to_numpy(dtype=str)
    if (node_ids == "").any() or len(set(node_ids)) != len(node_ids) or (labels == "").any():
        raise ValueError("Node IDs must be unique and IDs and labels must be nonempty")
    values = content.iloc[:, 1:-1].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Node features must be finite")
    features = sparse.csr_matrix(values)
    sums = np.abs(features).sum(axis=1).A1
    scale = np.divide(1, sums, out=np.zeros_like(sums), where=sums != 0)
    features = features.multiply(scale[:, None]).tocsr()

    index = {node_id: i for i, node_id in enumerate(node_ids)}
    links = []
    with Path(edges).open(encoding="utf-8-sig", newline="") as stream:
        for row_number, row in enumerate(csv.reader(stream, delimiter="\t"), 1):
            if not row:
                continue
            if len(row) != 2 or any(node_id not in index for node_id in row):
                raise ValueError(f"Edge row {row_number} needs two known node IDs")
            links.append((index[row[0]], index[row[1]]))
    source, target = np.array(links, dtype=int).reshape(-1, 2).T
    graph = sparse.csr_matrix((np.ones(len(links)), (source, target)), shape=(len(node_ids), len(node_ids)))
    graph = graph.maximum(graph.T)
    graph.setdiag(0)
    graph.eliminate_zeros()
    graph.data.fill(1.0)
    classes, encoded = np.unique(labels, return_inverse=True)
    return node_ids, classes, encoded, features, graph, len(links)


def split_nodes(labels, train_per_class, validation_fraction, seed):
    rng = np.random.default_rng(seed)
    partitions = {"train": [], "validation": [], "test": []}
    for label in np.unique(labels):
        nodes = rng.permutation(np.flatnonzero(labels == label))
        remaining = len(nodes) - train_per_class
        if remaining < 2:
            raise ValueError("Each class needs training nodes and at least two remaining nodes")
        validation_size = max(1, min(remaining - 1, round(remaining * validation_fraction)))
        cut = train_per_class + validation_size
        partitions["train"].extend(nodes[:train_per_class])
        partitions["validation"].extend(nodes[train_per_class:cut])
        partitions["test"].extend(nodes[cut:])
    return {name: np.sort(rows) for name, rows in partitions.items()}


def normalize_graph(graph):
    with_self = graph + sparse.eye(graph.shape[0], format="csr")
    scale = sparse.diags(1 / np.sqrt(with_self.sum(axis=1).A1), format="csr")
    return (scale @ with_self @ scale).tocsr()


def gcn_loss(messages, adjacency, weights, train, train_labels):
    logits, hidden, neighborhood = graph_forward(messages, adjacency, weights)
    labeled = logits[train]
    log_prob = labeled - logsumexp(labeled, axis=1, keepdims=True)
    row = np.arange(len(train))
    loss = -log_prob[row, train_labels].mean() + WEIGHT_DECAY / 2 * np.sum(weights[0]**2)
    derivative = np.exp(log_prob)
    derivative[row, train_labels] -= 1
    gradients = np.zeros_like(logits)
    gradients[train] = derivative / len(train)
    return loss, graph_gradients(messages, adjacency, weights, hidden, neighborhood, gradients), logits


def fit_gcn(features, graph, labels, partitions, *, seed, epochs, hidden_size):
    adjacency = normalize_graph(graph)
    train, validation = partitions["train"], partitions["validation"]
    classes = np.unique(labels[train])
    train_labels = labels[train]

    def validation_f1(logits):
        return metrics(labels[validation], logits[validation].argmax(axis=1), classes)["F1"]

    weights, epoch, history = train_graph_layers(
        features, adjacency, output_size=len(classes), seed=seed, epochs=epochs, hidden_size=hidden_size,
        objective=lambda x, a, w: gcn_loss(x, a, w, train, train_labels), validation_score=validation_f1,
    )
    return graph_forward(adjacency @ features, adjacency, weights)[0].argmax(axis=1), epoch, history


def fit_logistic(features, labels, partitions):
    train, validation = partitions["train"], partitions["validation"]
    classes = np.unique(labels[train])
    best_f1 = -1.0
    history = []
    for c in LOGISTIC_C:
        model = LogisticRegression(C=c, max_iter=1000).fit(features[train], labels[train])
        predictions = model.predict(features)
        f1 = metrics(labels[validation], predictions[validation], classes)["F1"]
        history.append((c, f1))
        if f1 > best_f1:
            best_f1, best_c, best_predictions = f1, c, predictions
    return best_predictions, best_c, history


def comparison_rows(labels, partitions, predictions):
    classes = np.unique(labels)
    return [{"Model": METHODS[method], "Split": part, **metrics(labels[rows], prediction[rows], classes)}
            for method, prediction in predictions.items()
            for part, rows in partitions.items() if part != "train"]


def classify_graph_nodes(nodes, edges, output, *, seed=42, train_per_class=20, validation_fraction=0.2,
                         hidden_size=16, epochs=120):
    """Compare a GCN with a feature-only baseline using labeled and held-out graph nodes."""
    integers = (seed, train_per_class, hidden_size, epochs)
    if (any(isinstance(value, bool) or not isinstance(value, Integral) for value in integers)
            or seed < 0 or min(integers[1:]) < 1 or not 0 < validation_fraction < 1):
        raise ValueError("Use a nonnegative seed, positive integer counts and a validation fraction between zero and one")
    settings = dict(nodes=str(nodes), edges=str(edges), seed=seed, train_per_class=train_per_class,
                    validation_fraction=validation_fraction, hidden_size=hidden_size, epochs=epochs)
    hashes = {key: file_sha256(settings[key]) for key in ("nodes", "edges")}
    node_ids, classes, labels, features, graph, edge_rows = read_graph(nodes, edges)
    if len(classes) < 2:
        raise ValueError("Node classification needs at least two classes")
    partitions = split_nodes(labels, train_per_class, validation_fraction, seed)
    if any(file_sha256(settings[key]) != digest for key, digest in hashes.items()):
        raise ValueError("Graph inputs changed while being read")
    output, manifest = prepare(output, SCHEMA, settings)
    try:
        started = perf_counter()
        logistic, best_c, logistic_history = fit_logistic(features, labels, partitions)
        logistic_time = perf_counter() - started
        started = perf_counter()
        gcn, best_epoch, gcn_history = fit_gcn(features, graph, labels, partitions, seed=seed,
                                             epochs=epochs, hidden_size=hidden_size)
        gcn_time = perf_counter() - started
        predictions = {"logistic": logistic, "gcn": gcn}
        comparison = comparison_rows(labels, partitions, predictions)
        selected = max((r for r in comparison if r["Split"] == "validation"), key=lambda r: r["F1"])["Model"]
        test = partitions["test"]
        sheets = {
            "Comparison": pd.DataFrame(comparison),
            "GCN_Training": pd.DataFrame(gcn_history, columns=["Epoch", "Training loss", "Validation macro F1"]),
            "Logistic_Validation": pd.DataFrame(logistic_history, columns=["C", "Validation macro F1"]),
            "Test_Predictions": pd.DataFrame({"Node ID": node_ids[test], "Label": classes[labels[test]],
                                              **{METHODS[m]: classes[p[test]] for m, p in predictions.items()}}),
            "Summary": summary([
                ("Protocol", "Transductive: all node features and edges visible; training uses only training labels"),
                ("Selection", "Validation macro F1; keep the first candidate on ties"), ("Selected", selected),
                ("Nodes", len(node_ids)), ("Features", features.shape[1]), ("Classes", len(classes)),
                ("Input edge rows", edge_rows), ("Undirected links", graph.nnz // 2),
                ("Isolated nodes", int(np.count_nonzero(graph.getnnz(axis=1) == 0))),
                *((part.capitalize() + " nodes", len(rows)) for part, rows in partitions.items()),
                ("Logistic C", best_c), ("GCN checkpoint epoch", best_epoch),
                ("Logistic training and selection seconds", logistic_time), ("GCN training and selection seconds", gcn_time),
            ]),
        }
        manifest.update(input_sha256=hashes, classes=classes, selected_model=selected,
                        model_settings={"logistic_c": best_c, "gcn_epoch": best_epoch,
                                        "gcn_learning_rate": LEARNING_RATE, "gcn_weight_decay": WEIGHT_DECAY})
        return save(output, manifest, sheets, dict(node_ids=node_ids, labels=labels, **partitions, **predictions))
    except Exception as error:
        fail(output, manifest, error)
        raise


def validate_graph_classification(manifest, arrays, sheets):
    from .ul_subsample_validation import same_table

    if set(sheets) != {"Comparison", "GCN_Training", "Logistic_Validation", "Test_Predictions", "Summary"}:
        raise ValueError("Graph report sheets differ")
    labels, node_ids = arrays["labels"], arrays["node_ids"]
    classes = np.asarray(manifest["classes"], dtype=str)
    if (labels.ndim != 1 or labels.dtype.kind not in "iu" or node_ids.shape != labels.shape
            or node_ids.dtype.kind != "U" or len(np.unique(node_ids)) != len(labels)
            or len(np.unique(classes)) < 2 or not np.array_equal(np.unique(labels), np.arange(len(classes)))):
        raise ValueError("Graph labels or node IDs differ")
    settings = manifest["settings"]
    partitions = split_nodes(labels, settings["train_per_class"], settings["validation_fraction"], settings["seed"])
    if any(not np.array_equal(arrays[part], rows) for part, rows in partitions.items()):
        raise ValueError("Graph node partitions differ")
    predictions = {method: arrays[method] for method in METHODS}
    for values in predictions.values():
        if (values.shape != labels.shape or values.dtype.kind not in "iu"
                or (values < 0).any() or (values >= len(classes)).any()):
            raise ValueError("Invalid graph predictions")
    comparison = pd.DataFrame(comparison_rows(labels, partitions, predictions))
    same_table(sheets["Comparison"], comparison, "Graph classification metrics differ")
    validation = comparison.query("Split == 'validation'").set_index("Model")["F1"]
    if manifest["selected_model"] != validation.idxmax():
        raise ValueError("Graph model selection differs")
    test = partitions["test"]
    actual = pd.DataFrame({"Node ID": node_ids[test], "Label": classes[labels[test]],
                           **{METHODS[m]: classes[p[test]] for m, p in predictions.items()}})
    same_table(sheets["Test_Predictions"], actual, "Graph test predictions differ")
    selected = manifest["model_settings"]
    for sheet, column, candidates, method, choice in (
        ("GCN_Training", "Epoch", np.arange(1, settings["epochs"] + 1), "gcn", "gcn_epoch"),
        ("Logistic_Validation", "C", LOGISTIC_C, "logistic", "logistic_c"),
    ):
        history = sheets[sheet]
        scores = history["Validation macro F1"].to_numpy(dtype=float)
        if not np.array_equal(history[column], candidates) or not np.isfinite(scores).all():
            raise ValueError(f"Invalid graph validation history: {sheet}")
        if selected[choice] != candidates[scores.argmax()] or not np.isclose(scores.max(), validation[METHODS[method]]):
            raise ValueError(f"Graph validation choice differs: {sheet}")
