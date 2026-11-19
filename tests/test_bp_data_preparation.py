import csv

import pytest
from openpyxl import load_workbook

from diy_automl import prepare_bp_csv
from diy_automl.bp_preview import projection_features, run_projection, save_audit
from diy_automl.data_audit_validation import validate_data_audit
from diy_automl.data_extract_validation import validate_data_extract
from diy_automl.reporting import file_sha256


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


@pytest.mark.parametrize("compress", [False, True])
def test_bp_compares_a_projected_csv_and_keeps_the_audited_input(tmp_path, compress):
    if compress:
        pytest.importorskip("pyarrow")
        pytest.importorskip("duckdb")
    source = tmp_path / "input.csv"
    source.write_text("分類,値,補助列\nA,001,\nB,001,9\nB,002,0\n", encoding="utf-8")
    digest = file_sha256(source)
    original = save_audit(source, tmp_path / "original", target="分類", compress=compress)
    assert projection_features(original.profile) == ["値", "補助列"]
    selected = run_projection(original, ["値"])
    extracted = validate_data_extract(selected.csv.parent)
    assert extracted["settings"]["columns"] == ["分類", "値"]
    if compress:
        assert extracted["source_audit"]["original_csv_sha256"] == digest
    else:
        assert extracted["input_sha256"] == digest
    validate_data_audit(selected.audit.directory, csv=selected.csv)
    before, after = original.profile, selected.audit.profile
    assert before["target"] == after["target"]
    assert before["duplicates"]["features"]["conflicting_label_groups"] == 0
    assert after["duplicates"]["features"]["conflicting_label_groups"] == 1
    assert first_sheet_values(original.directory)["欠損セル数"] == 1
    assert first_sheet_values(selected.audit.directory)["欠損セル数"] == 0
    with selected.csv.open(encoding="utf-8", newline="") as stream:
        assert list(csv.reader(stream)) == [["分類", "値"], ["A", "001"], ["B", "001"], ["B", "002"]]
    assert file_sha256(source) == digest

    with source.open("a", encoding="utf-8") as stream:
        stream.write("A,003,4\n")
    with pytest.raises(ValueError, match="元CSVが点検後に変更|no longer matches"):
        run_projection(original, ["値"])
    assert selected.csv.is_file()
