from hashlib import sha256
import json

from openpyxl import load_workbook
import pytest

from diy_automl import audit_data_csv
from diy_automl.data_audit import SCHEMA
from diy_automl.data_audit_validation import validate_data_audit
from diy_automl.validation import validate


def read_profile(output):
    return json.loads((output / "profile.json").read_text(encoding="utf-8"))


def test_audit_bad_values_duplicates_and_context_without_fitting(tmp_path, monkeypatch):
    source = tmp_path / "data.csv"
    original = (
        "id,time,measure,category,target\n"
        "001,2026-11-18T10:00:00+09:00,1,,A\n"
        "001,2026-11-18T10:00:00+09:00,1,,A\n"
        "002,2026-11-18T05:00:00Z,1,,B\n"
        ",not-a-date,1e309,steel,B\n"
        "003,2026-11-18T01:30:00Z,2.5,x,\n"
        "short,2026-11-19\n"
    ).encode("utf-8")
    source.write_bytes(original)

    def unexpected_fit(*_args, **_kwargs):
        pytest.fail("Audit must not fit a model")

    monkeypatch.setattr("sklearn.pipeline.Pipeline.fit", unexpected_fit)
    output = audit_data_csv(
        source, tmp_path / "audit", target="target", id_column="id", time_column="time",
        row_unit="穀粒📏", prediction_goal="=1+1",
    )
    manifest = validate(output, SCHEMA)
    validate_data_audit(output, csv=source)
    assert source.read_bytes() == original
    assert manifest["input_sha256"] == sha256(original).hexdigest()

    profile = read_profile(output)
    assert profile["scan"]["valid_rows"] == 5
    assert profile["scan"]["wrong_width_examples"] == [{"line": 7, "fields": 2}]
    measure, category = profile["columns"][2:4]
    assert (measure["nonfinite"], measure["numeric_max"]) == (1, 2.5)
    assert (category["missing"], category["observed_type"]) == (3, "text")
    assert profile["duplicates"]["whole_rows"]["extra_rows"] == 1
    assert profile["duplicates"]["features"]["extra_rows"] == 2
    assert profile["duplicates"]["features"]["conflicting_label_groups"] == 1
    assert profile["target"]["classes"] == [{"label": "A", "rows": 2}, {"label": "B", "rows": 2}]
    assert (profile["target"]["missing"], profile["target"]["classes_under_six"]) == (1, 2)
    assert (profile["id"]["distinct"], profile["id"]["repeated_ids"], profile["id"]["missing"]) == (3, 1, 1)
    assert profile["time"]["earliest"] == "2026-11-18T01:00:00+00:00"
    assert profile["time"]["latest"] == "2026-11-18T05:00:00+00:00"
    assert profile["time"]["invalid"] == 1

    workbook = load_workbook(output / "report.xlsx")
    try:
        assert workbook["Evaluation_Context"]["B2"].value == "=1+1"
        assert workbook["Evaluation_Context"]["B2"].data_type == "s"
    finally:
        workbook.close()
    source.write_bytes(original + b"004,,0,,B\n")
    with pytest.raises(ValueError, match="Input CSV checksum"):
        validate_data_audit(output, csv=source)


@pytest.mark.parametrize(("content", "state", "valid_rows"), [
    (b"\n", "no_header", 0),
    (b"x,target\n1,A\n2,\"unterminated\n", "partial", 1),
    (b"x,target\n1,\xff\n", "partial", 0),
    (b"x,x,target\n1,2,A\n", "complete", 1),
])
def test_untrainable_csv_still_writes_a_valid_report(tmp_path, content, state, valid_rows):
    source = tmp_path / "input.csv"
    source.write_bytes(content)
    output = audit_data_csv(source, tmp_path / "audit", target="target")
    assert validate_data_audit(output, csv=source)["scan_status"] == state
    profile = read_profile(output)
    assert profile["scan"]["valid_rows"] == valid_rows
    assert profile["id"]["status"] == "not_declared"
    assert profile["id"]["distinct"] is None
    assert profile["time"]["earliest"] is None
    codes = {issue["code"] for issue in profile["issues"]}
    assert {"incomplete_input"} <= codes if state != "complete" else "duplicate_header" in codes
