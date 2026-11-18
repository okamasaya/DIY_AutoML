import csv
from decimal import Decimal
import json
from pathlib import Path

import pytest

from diy_automl import audit_data_csv, extract_data
from diy_automl.data_extract import SCHEMA, bound_filter
from diy_automl.data_extract_validation import validate_data_extract
from diy_automl.reporting import file_sha256
from diy_automl.validation import validate


@pytest.fixture
def stored_data(tmp_path):
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")
    source = tmp_path / "source.csv"
    title = '測定 値 "cm"'
    with source.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerows([
            ["ID", title, "target"], ["001", "1.25", "A"], ["002", "2.75", "B"],
            ["003", "3.00", "B"], ["004", "4.50", "A"],
        ])
    audit = audit_data_csv(source, tmp_path / "kind=sample", target="target", id_column="ID",
                           convert=("parquet", "duckdb"), conversion_types="numeric")
    return {"title": title, "root": audit, "original": source}


@pytest.mark.parametrize("form", ["parquet", "duckdb"])
def test_extract_variable_columns_filters_and_limit(tmp_path, stored_data, form):
    audit = stored_data["root"]
    source = audit / f"converted/data.{form}"
    digest = file_sha256(source)
    output = extract_data(
        source, tmp_path / "extract", ["target", "ID"],
        where=((stored_data["title"], "ge", Decimal("2.75")), ("target", "eq", "B")), limit=1,
    )
    manifest = validate(output, SCHEMA)
    validate_data_extract(output, source=source)
    assert manifest["rows"] == 1
    assert manifest["input_sha256"] == digest == file_sha256(source)
    assert manifest["source_audit"]["audit_manifest_sha256"] == file_sha256(audit / "manifest.json")
    assert manifest["source_audit"]["original_csv_sha256"] == file_sha256(stored_data["original"])
    with (output / "data.csv").open(encoding="utf-8", newline="") as stream:
        assert list(csv.reader(stream)) == [["target", "ID"], ["B", "002"]]

    empty = extract_data(source, tmp_path / "header", columns=[stored_data["title"], "target"], limit=0)
    assert validate_data_extract(empty, source=source)["rows"] == 0


def test_extract_rejects_unknown_columns_and_imprecise_filters(tmp_path, stored_data):
    source = stored_data["root"] / "converted/data.parquet"
    for index, (columns, where, message) in enumerate([
        (["not_in_file"], (), "Columns not found"),
        (["target"], (("ID", "ge", "002"),), "Text filters use eq or ne"),
        (["target"], ((stored_data["title"], "ge", "2.751"),), "does not fit DECIMAL"),
    ]):
        output = tmp_path / f"rejected-{index}"
        with pytest.raises(ValueError, match=message):
            extract_data(source, output, columns=columns, where=where)
        assert not (output / "data.csv").exists()
        assert json.loads((output / "manifest.json").read_text(encoding="utf-8"))["status"] == "failed"


def test_extract_detects_a_modified_audited_file(tmp_path, stored_data):
    source = stored_data["root"] / "converted/data.parquet"
    with source.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="does not match its conversion manifest"):
        extract_data(source, tmp_path / "extract", columns=["target"])
    assert not (tmp_path / "extract").exists()


def test_extract_requires_a_duckdb_base_table(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    source = Path(tmp_path / "source.duckdb")
    with duckdb.connect(str(source)) as db:
        db.execute("CREATE VIEW data AS SELECT 1 AS value")
    with pytest.raises(ValueError, match="base table main.data"):
        extract_data(source, tmp_path / "extract", columns=["value"])
    assert not (tmp_path / "extract/data.csv").exists()


def test_integer_filter_validates_its_range_before_querying():
    assert bound_filter("BIGINT", "ge", "3.000", "$value") == ("CAST($value AS BIGINT)", "3")
    with pytest.raises(ValueError, match="does not fit BIGINT"):
        bound_filter("BIGINT", "eq", str(2**63), "$value")
    with pytest.raises(ValueError, match="does not fit UBIGINT"):
        bound_filter("UBIGINT", "eq", "-1", "$value")
