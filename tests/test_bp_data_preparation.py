import csv

import pytest
from openpyxl import load_workbook

from diy_automl import prepare_bp_csv
from diy_automl.data_audit_validation import validate_data_audit


def first_sheet_values(output):
    workbook = load_workbook(output / "report.xlsx", read_only=True)
    try:
        assert workbook.sheetnames[0] == "まず読む"
        return dict(workbook["まず読む"].values)
    finally:
        workbook.close()


def test_bp_audit_without_compression_uses_the_existing_core(tmp_path, monkeypatch):
    def unexpected_conversion(*args):
        pytest.fail("Compression must be opt-in")

    monkeypatch.setattr("diy_automl.data_audit.convert_csv", unexpected_conversion)
    source = tmp_path / "input.csv"
    source.write_text("測定値,分類\n10,A\n20,B\n", encoding="utf-8")
    output = prepare_bp_csv(source, tmp_path / "report", target="分類")
    manifest = validate_data_audit(output, csv=source)
    assert set(manifest["artifacts"]) == {"report.xlsx", "profile.json"}
    summary = first_sheet_values(output)
    assert summary["作業用ファイルを圧縮"] == "指定なし"
    assert summary["列数が合ったデータ行"] == 2
    assert summary["正解ラベルの種類数"] == 2


def test_bp_compression_has_size_summary_and_keeps_the_original(tmp_path):
    parquet = pytest.importorskip("pyarrow.parquet")
    source = tmp_path / "input.csv"
    rows = [
        {"商品ID": "001", "価格": "1.20", "分類": "A"},
        {"商品ID": "002", "価格": "NA", "分類": "B"},
    ]
    with source.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    original = source.read_bytes()
    output = prepare_bp_csv(source, tmp_path / "report", target="分類", compress=True)
    manifest = validate_data_audit(output, csv=source)
    assert manifest["settings"]["conversion_types"] == "preserve"
    archive = output / "converted/data.parquet"
    assert parquet.read_table(archive).to_pylist() == rows
    with parquet.ParquetFile(archive) as saved:
        assert saved.metadata.row_group(0).column(0).compression == "ZSTD"
    assert source.read_bytes() == original
    summary = first_sheet_values(output)
    assert summary["作業用ファイルを圧縮"] == "保存済み"
    assert summary["元CSVのサイズ（バイト）"] == len(original)
    assert summary["圧縮後のサイズ（バイト）"] == archive.stat().st_size
    assert summary["圧縮後のサイズ（元CSV比 %）"] == round(100 * archive.stat().st_size / len(original), 1)
    assert not (output / "converted/data.duckdb").exists()


def test_bp_keeps_the_japanese_report_when_compression_is_blocked(tmp_path):
    source = tmp_path / "input.csv"
    source.write_text("x,target\n1,A\nshort\n", encoding="utf-8")
    output = prepare_bp_csv(source, tmp_path / "report", target="target", compress=True)
    manifest = validate_data_audit(output, csv=source)
    assert manifest["conversion"]["status"] == "blocked"
    summary = first_sheet_values(output)
    assert summary["作業用ファイルを圧縮"] == "元データの確認が必要"
    assert summary["列数が合わない行"] == 1
    assert not (output / "converted").exists()
