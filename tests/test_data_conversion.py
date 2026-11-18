import csv
from decimal import Decimal
import json

import pytest

from diy_automl import audit_data_csv
from diy_automl.data_audit_validation import validate_data_audit
from diy_automl.data_conversion import CONVERSION_SCHEMA
from diy_automl.validation import validate


@pytest.mark.parametrize("mode", ["preserve", "numeric"])
def test_storage_round_trip_keeps_values_and_protected_columns(tmp_path, mode):
    duckdb = pytest.importorskip("duckdb")
    parquet = pytest.importorskip("pyarrow.parquet")
    source = tmp_path / "input.csv"
    names = ["id", "time", "target", 'quoted "number"', "precise", "missing", "padded", "note"]
    rows = [
        ["001", "2026-11-18", "01", "1", "1.20", "", "007", "first\nline"],
        ["002", "2026-11-19", "02", "+2", "0.123456789012345678901234567890", "NA", "008", "日本語"],
        ["003", "2026-11-19", "01", "3e0", "12345678.000", "NULL", "019", 'has, a "quote"'],
    ]
    with source.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(names)
        writer.writerows(rows)
    original = source.read_bytes()
    output = audit_data_csv(source, tmp_path / "audit", target="target", id_column="id", time_column="time",
                            convert=("parquet", "duckdb"), conversion_types=mode)
    manifest = validate(output, CONVERSION_SCHEMA)
    validate_data_audit(output, csv=source)
    assert source.read_bytes() == original
    assert manifest["conversion"]["status"] == "completed"
    assert not list(output.glob(".conversion-*"))
    if mode == "numeric":
        for index, row in enumerate(rows, start=1):
            row[3], row[4] = index, Decimal(row[4])
        assert [item["type"] for item in manifest["conversion"]["columns"]] == [
            "VARCHAR", "VARCHAR", "VARCHAR", "BIGINT", "DECIMAL(38,30)", "VARCHAR", "VARCHAR", "VARCHAR",
        ]
    parquet_path = output / "converted/data.parquet"
    actual = parquet.read_table(parquet_path)
    assert actual.column_names == names
    assert actual.to_pylist() == [dict(zip(names, row)) for row in rows]
    with duckdb.connect(str(output / "converted/data.duckdb"), read_only=True) as conn:
        assert conn.execute("SELECT * FROM data").fetchall() == [tuple(row) for row in rows]
    with parquet_path.open("ab") as stream:
        stream.write(b"altered")
    with pytest.raises(ValueError, match="Artifact checksum mismatch"):
        validate_data_audit(output)


@pytest.mark.parametrize(("data", "reason"), [
    ("x,target\n1,A\nextra\n", "Repair the CSV structure"),
    ("x,X,target\n1,2,A\n", "distinct ignoring ASCII case"),
])
def test_blocked_conversion_preserves_the_audit(tmp_path, data, reason):
    source = tmp_path / "input.csv"
    source.write_text(data, encoding="utf-8")
    output = audit_data_csv(source, tmp_path / "audit", target="target", convert="duckdb")
    manifest = validate_data_audit(output, csv=source)
    assert manifest["conversion"]["status"] == "blocked"
    assert reason in manifest["conversion"]["reason"]
    assert set(manifest["artifacts"]) == {"profile.json", "report.xlsx"}
    assert not (output / "converted").exists()


def test_failed_conversion_never_publishes_partial_files(tmp_path, monkeypatch):
    source = tmp_path / "input.csv"
    source.write_text("x,target\n1,A\n", encoding="utf-8")

    def failed_writer(_csv, _stage, files, _columns, _profile):
        files["parquet"].write_bytes(b"incomplete output")
        raise RuntimeError("Writer failed")

    monkeypatch.setattr("diy_automl.data_conversion.write_formats", failed_writer)
    output = audit_data_csv(source, tmp_path / "audit", target="target", convert=("parquet", "duckdb"))
    manifest = validate_data_audit(output, csv=source)
    assert manifest["conversion"]["status"] == "failed"
    assert "Writer failed" in manifest["conversion"]["reason"]
    assert not (output / "converted").exists()
    assert not list(output.glob(".conversion-*"))
    profile = json.loads((output / "profile.json").read_text(encoding="utf-8"))
    assert profile["scan"]["valid_rows"] == 1
