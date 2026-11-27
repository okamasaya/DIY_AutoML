import numpy as np
import pytest
from scipy import sparse

from diy_automl import classify_graph_nodes
from diy_automl.graph_classification import SCHEMA, fit_gcn, gcn_loss, normalize_graph, read_graph
from diy_automl.validation import validate


def test_sparse_gcn_gradients_match_finite_differences():
    rng = np.random.default_rng(42)
    graph = sparse.csr_matrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    adjacency = normalize_graph(graph)
    messages = adjacency @ sparse.csr_matrix(rng.normal(size=(3, 2)))
    weights = [rng.normal(size=(2, 3)), rng.normal(size=(3, 2))]
    train, labels = np.array([0, 2]), np.array([1, 0])
    _, gradients, _ = gcn_loss(messages, adjacency, weights, train, labels)
    epsilon = 1e-6
    for weight, gradient in zip(weights, gradients, strict=True):
        expected = np.empty_like(gradient)
        for index in np.ndindex(weight.shape):
            original = weight[index]
            weight[index] = original + epsilon
            greater = gcn_loss(messages, adjacency, weights, train, labels)[0]
            weight[index] = original - epsilon
            smaller = gcn_loss(messages, adjacency, weights, train, labels)[0]
            weight[index] = original
            expected[index] = (greater - smaller) / (2 * epsilon)
        np.testing.assert_allclose(gradient, expected, atol=2e-9)


def test_holdout_labels_do_not_affect_gcn_fit_or_checkpoint():
    features = sparse.csr_matrix([[1, 0], [0, 1]] * 3, dtype=float)
    graph = sparse.diags([np.ones(5), np.ones(5)], [-1, 1], format="csr")
    partitions = {"train": np.array([0, 1]), "validation": np.array([2, 3]), "test": np.array([4, 5])}
    labels = np.array([0, 1] * 3)
    altered = labels.copy()
    altered[partitions["test"]] = 1
    options = dict(seed=42, epochs=5, hidden_size=4)
    baseline = fit_gcn(features, graph, labels, partitions, **options)
    changed = fit_gcn(features, graph, altered, partitions, **options)
    np.testing.assert_array_equal(baseline[0], changed[0])
    assert baseline[1:] == changed[1:]


def test_graph_workflow_uses_literal_ids_and_validates_saved_results(tmp_path):
    nodes, edges = tmp_path / "nodes.tsv", tmp_path / "edges.tsv"
    nodes.write_text("001\t1\t0\tNA\n002\t1\t0\tNA\n003\t1\t0\tNA\n"
                     "004\t0\t1\tB\n005\t0\t1\tB\n006\t0\t1\tB\n", encoding="utf-8")
    edges.write_text("001\t002\n002\t001\n001\t001\n004\t005\n", encoding="utf-8")
    _, _, _, _, graph, edge_rows = read_graph(nodes, edges)
    assert edge_rows == 4 and graph.nnz == 4
    assert normalize_graph(graph)[2, 2] == 1

    result = classify_graph_nodes(nodes, edges, tmp_path / "result", train_per_class=1, epochs=5)
    assert validate(result, SCHEMA)["classes"] == ["B", "NA"]
    with np.load(result / "arrays.npz", allow_pickle=False) as arrays:
        assert arrays["node_ids"].tolist() == ["001", "002", "003", "004", "005", "006"]
        assert sorted(np.concatenate([arrays[name] for name in ("train", "validation", "test")])) == list(range(6))

    edges.write_text("001\tmissing\n", encoding="utf-8")
    with pytest.raises(ValueError, match="known node IDs"):
        classify_graph_nodes(nodes, edges, tmp_path / "invalid", train_per_class=1)
    assert not (tmp_path / "invalid").exists()
