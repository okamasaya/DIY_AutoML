import csv
import json

import numpy as np
import pytest
from openpyxl import load_workbook

from diy_automl.bp_evaluation import BP_REPORT, evaluate_selection, training_blockers
from diy_automl.bp_preview import run_projection, save_audit
from diy_automl.selection_workflow import select_csv
from diy_automl.validation import validate


def test_bp_comparison_preserves_labels_and_explains_the_same_holdout(tmp_path):
    source = tmp_path / "original.csv"
    with source.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["値", "正解"])
        for index, label in enumerate(("001", "1", "=1+1")):
            writer.writerows([(index, label)] * 12)
    audit = save_audit(source, tmp_path / "audit", "正解")
    selection = run_projection(audit, ["値"])
    # Duplicate inputs are evidence to show, and are tracked on the actual split.
    assert not training_blockers(selection.audit.profile)
    output = select_csv(selection.csv, "正解", tmp_path / "evaluation", bootstrap_repeats=30,
                        expected_input_sha256=selection.audit.profile["input_sha256"], bp_summary=True)
    manifest = validate(output, 3)
    assert manifest["classes"] == ["001", "1", "=1+1"]
    result = manifest["bp_summary"]
    with np.load(output / "arrays.npz", allow_pickle=False) as arrays:
        truth, predictions = arrays["truth"], arrays["predictions"]
        assert result["test_rows"] == len(truth) == result["test_rows_also_in_train"]
        assert result["correct"] == int((truth == predictions).sum())
        assert result["correct"] + result["incorrect"] == len(truth)
    assert sum(row[2] for row in result["classes"]) == result["correct"]
    assert sum(row[3] for row in result["classes"]) == sum(row[4] for row in result["classes"]) == result["incorrect"]
    assert result["majority_correct"] == next(row[1] for row in result["classes"] if row[0] == result["majority_label"])

    book = load_workbook(output / BP_REPORT, read_only=True, data_only=False)
    try:
        assert book.sheetnames[:2] == ["まず読む", "ラベル別"]
        assert book["ラベル別"]["A4"].value == "=1+1"
        assert book["ラベル別"]["A4"].data_type == "s"
        assert len(list(book["SL_CV_Mean"].values)) == 6
        overview = dict(book["まず読む"].values)
        assert overview["正解"] == result["correct"]
        assert overview["95%区間の下端"] == pytest.approx(result["f1_lower"])
    finally:
        book.close()


def test_bp_comparison_requires_the_audited_snapshot(tmp_path):
    source = tmp_path / "original.csv"
    source.write_text("x,label\n" + "1,A\n2,B\n" * 12, encoding="utf-8")
    audit = save_audit(source, tmp_path / "audit", "label")
    selection = run_projection(audit, ["x"])
    with selection.csv.open("a", encoding="utf-8") as stream:
        stream.write("3,A\n")
    output = tmp_path / "evaluation"
    with pytest.raises(ValueError, match="no longer matches its audit"):
        evaluate_selection(selection, output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert not (output / BP_REPORT).exists()
