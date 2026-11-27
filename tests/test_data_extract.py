import csv
from decimal import Decimal
import json
from pathlib import Path

import pytest

from diy_automl import audit_data_csv, extract_data
from diy_automl.cli import main
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


def test_csv_projection_preserves_cell_text_and_order_without_storage(tmp_path):
    source = tmp_path / "source.csv"
    with source.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerows([
            [], ["unused", "値", "label"], ["0", "001", "甲"],
            [], ["1", '1.20,\r\n"引用"', "乙"], ["2", "", "甲"],
        ])
    digest = file_sha256(source)
    output = extract_data(source, tmp_path / "extract", ["label", "値"], limit=2)
    manifest = validate_data_extract(output, source=source)
    assert manifest["duckdb_version"] is None
    assert manifest["rows"] == 2
    assert manifest["input_sha256"] == digest == file_sha256(source)
    with (output / "data.csv").open(encoding="utf-8", newline="") as stream:
        assert list(csv.reader(stream)) == [["label", "値"], ["甲", "001"], ["乙", '1.20,\r\n"引用"']]


def test_csv_projection_checks_rows_after_the_limit(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("x,target\n1,A\nbroken\n", encoding="utf-8")
    output = tmp_path / "extract"
    with pytest.raises(ValueError, match="record width differs near line 3"):
        extract_data(source, output, ["target"], limit=1)
    assert not (output / "data.csv").exists()


@pytest.mark.parametrize("form", ["csv", "parquet", "duckdb"])
def test_cli_recipe_keeps_only_selected_format_and_exact_csv_cells(tmp_path, monkeypatch, form):
    if form != "csv":
        pytest.importorskip("pyarrow")
    if form == "duckdb":
        pytest.importorskip("duckdb")
    source = tmp_path / "source.csv"
    source.write_text("Label,ID,Unused\nA,001,x\nB,,y\nA,3.20,z\n", encoding="utf-8")
    digest = file_sha256(source)
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({"columns": ["ID", "Label"], "format": form}), encoding="utf-8-sig")
    output = tmp_path / "extract"
    monkeypatch.setattr("sys.argv", ["diy-automl", "extract-data", "--source", str(source),
                                    "--recipe", str(recipe), "--output", str(output)])
    main()
    manifest = validate_data_extract(output, source)
    assert manifest["input_sha256"] == digest == file_sha256(source)
    assert {path.name for path in output.iterdir()} == {"manifest.json", f"data.{form}"}
    if form == "parquet":
        from pyarrow.parquet import read_table

        rows = [tuple(row.values()) for row in read_table(output / "data.parquet").to_pylist()]
    elif form == "duckdb":
        import duckdb

        with duckdb.connect(str(output / "data.duckdb"), read_only=True) as db:
            rows = db.execute("SELECT * FROM main.data").fetchall()
    else:
        with (output / "data.csv").open(encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream)
            assert next(reader) == ["ID", "Label"]
            rows = [tuple(row) for row in reader]
    assert rows == [("001", "A"), ("", "B"), ("3.20", "A")]


def test_duckdb_projection_preserves_null_and_native_integer_type(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    source = tmp_path / "source's.duckdb"
    with duckdb.connect(str(source)) as db:
        db.execute("CREATE TABLE data (wide HUGEINT, label VARCHAR)")
        db.execute("INSERT INTO data VALUES (CAST(? AS HUGEINT), NULL), (10, ''), (1, 'removed')", [str(2**126 + 1)])
    output = extract_data(source, tmp_path / "extract", ["label", "wide"],
                          where=(("wide", "ge", 10),), output_format="duckdb")
    assert validate_data_extract(output, source)["rows"] == 2
    with duckdb.connect(str(output / "data.duckdb"), read_only=True) as db:
        assert db.execute("SELECT * FROM main.data").fetchall() == [(None, 2**126 + 1), ("", 10)]
        assert db.execute("DESCRIBE SELECT wide FROM main.data").fetchone()[1] == "HUGEINT"

    rejected = tmp_path / "lossy"
    with pytest.raises(ValueError, match="Parquet would change"):
        extract_data(source, rejected, ["wide"], output_format="parquet")
    assert not (rejected / "data.parquet").exists()
