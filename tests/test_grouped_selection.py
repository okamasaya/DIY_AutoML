import csv
import json

import numpy as np
import pytest

from diy_automl.bp_evaluation import evaluate_selection
from diy_automl.bp_preview import projection_features, run_projection, save_audit
from diy_automl.data import load_csv
from diy_automl.grouped_selection import identical_input_groups
from diy_automl.reporting import file_sha256, write_json
from diy_automl.selection import bootstrap
from diy_automl.selection_workflow import select_csv
from diy_automl.supervised import metrics
from diy_automl.validation import validate


def test_grouped_bp_selection_keeps_ids_out_of_features_and_partitions(tmp_path):
    source = tmp_path / "data.csv"
    ids = ["001", "1"] + [f"ID{i}" for i in range(16)]
    with source.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["顧客ID", "値", "正解"])
        writer.writerows((identifier, index + offset, label)
                         for index, identifier in enumerate(ids)
                         for offset, label in [(0, "A"), (50, "B")])
    audit = save_audit(source, tmp_path / "original", "正解", id_column="顧客ID")
    assert projection_features(audit.profile) == ["値"]
    selected = run_projection(audit, ["値"])
    assert selected.audit.profile["id"]["distinct"] == len(ids)
    directory, result = evaluate_selection(selected, tmp_path / "evaluation")
    manifest = validate(directory, 3)
    assert manifest["features"] == ["値"]
    assert manifest["settings"]["group_column"] == "顧客ID"
    assert result["split"]["mode"] == "group_column"
    assert result["split"]["train_groups"] + result["split"]["test_groups"] == len(ids)

    with np.load(directory / "arrays.npz", allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    groups = arrays["group_ids"]
    assert groups[0] == groups[1] and groups[0] != groups[2]
    train, test = arrays["train"], arrays["test"]
    assert set(groups[train]).isdisjoint(groups[test])
    for fold in range(5):
        assert set(groups[arrays[f"fold_{fold}_train"]]).isdisjoint(groups[arrays[f"fold_{fold}_validation"]])

    groups[test[0]] = groups[train[0]]
    np.savez_compressed(directory / "arrays.npz", **arrays)
    manifest["artifacts"]["arrays.npz"] = file_sha256(directory / "arrays.npz")
    write_json(directory / "manifest.json", manifest)
    with pytest.raises(ValueError, match="Train/test group overlap"):
        validate(directory, 3)


def test_identical_numeric_features_are_kept_together_even_if_labels_differ(tmp_path):
    source = tmp_path / "data.csv"
    with source.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["x", "label"])
        writer.writerows((repr(float(index)) if label == "B" else index, label)
                         for index in range(25) for label in ["A", "B"])
    output = select_csv(source, "label", tmp_path / "evaluation", group_identical=True, bootstrap_repeats=25)
    manifest = validate(output, 3)
    assert manifest["split"]["mode"] == "identical_inputs"
    with np.load(output / "arrays.npz", allow_pickle=False) as arrays:
        groups = arrays["group_ids"]
        np.testing.assert_array_equal(groups, np.repeat(np.arange(25), 2))
        assert set(groups[arrays["train"]]).isdisjoint(groups[arrays["test"]])
    np.testing.assert_array_equal(identical_input_groups(np.array([[-0.0], [0.0], [1.0]])), [0, 0, 1])


def test_group_bootstrap_matches_resampling_entire_groups():
    truth = np.array([0, 1, 1, 0, 0, 0])
    predicted = np.array([0, 0, 1, 1, 0, 0])
    codes = np.array([7, 7, 5, 2, 2, 2])
    repeats, seed = 20, 15
    observed, _ = bootstrap(truth, predicted, repeats=repeats, seed=seed, group_ids=codes)
    rng = np.random.default_rng(seed)
    members = [np.flatnonzero(codes == group) for group in np.unique(codes)]
    expected = []
    for _ in range(repeats):
        indices = np.concatenate([members[group] for group in rng.integers(len(members), size=len(members))])
        expected.append(list(metrics(truth[indices], predicted[indices], [0, 1]).values()))
    np.testing.assert_allclose(observed.to_numpy(), expected)


def test_grouped_selection_requires_group_coverage_and_rejects_missing_ids(tmp_path):
    source = tmp_path / "data.csv"
    source.write_text("ID,x,label\n" + "same,1,A\nsame,2,B\n" * 9, encoding="utf-8")
    output = tmp_path / "not-enough"
    with pytest.raises(ValueError, match="at least six groups"):
        select_csv(source, "label", output, group_column="ID")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    source.write_text("ID,x,label\n,1,A\none,2,B\n", encoding="utf-8")
    with pytest.raises(ValueError, match="identifiers cannot be missing"):
        load_csv(source, "label", group_column="ID")
