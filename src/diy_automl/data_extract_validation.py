"""Check a projected data file and optionally confirm its input snapshot."""

from contextlib import closing
import csv
import json
from pathlib import Path

from .data_conversion import DUCKDB_CONFIG
from .data_extract import SCHEMA, extraction_settings
from .reporting import file_sha256


def validate_output(data, output_format, columns):
    if output_format == "parquet":
        from pyarrow.parquet import ParquetFile

        with closing(ParquetFile(data)) as stored:
            names, rows = stored.schema_arrow.names, stored.metadata.num_rows
    elif output_format == "duckdb":
        import duckdb

        config = {**DUCKDB_CONFIG, "enable_external_access": "false"}
        with closing(duckdb.connect(str(data), read_only=True, config=config)) as db:
            table = db.execute("""
                SELECT table_type FROM information_schema.tables
                WHERE table_catalog = current_database() AND table_schema = 'main' AND table_name = 'data'
            """).fetchone()
            if table != ("BASE TABLE",):
                raise ValueError("Extracted DuckDB must contain the base table main.data")
            names = [row[0] for row in db.execute("DESCRIBE SELECT * FROM main.data").fetchall()]
            rows = db.execute("SELECT count(*) FROM main.data").fetchone()[0]
    else:
        with data.open(encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream, strict=True)
            names = next(reader, None)
            rows = 0
            for row in reader:
                if len(row) != len(columns):
                    raise ValueError("Extracted CSV record width differs")
                rows += 1
    if names != columns:
        raise ValueError("Extracted columns differ")
    return rows


def validate_data_extract(directory, source=None):
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != SCHEMA or manifest["status"] != "completed":
        raise ValueError("Data extraction did not complete")
    settings = manifest["settings"]
    if settings != extraction_settings(**settings):
        raise ValueError("Extraction settings differ")
    output_format = settings.get("output_format", "csv")
    artifact = f"data.{output_format}"
    if set(manifest["artifacts"]) != {artifact}:
        raise ValueError("Unexpected extraction artifact set")
    data = root / artifact
    if file_sha256(data) != manifest["artifacts"][artifact]:
        raise ValueError("Extracted file checksum differs")
    source_columns = {column["name"] for column in manifest["source_columns"]}
    requested = set(settings["columns"]) | {term[0] for term in settings["where"]}
    if not requested <= source_columns:
        raise ValueError("Selected or filtered column is absent from source schema")
    if manifest["source_audit"] and manifest["source_audit"]["columns"] != manifest["source_columns"]:
        raise ValueError("Recorded columns differ from the source audit")
    rows = validate_output(data, output_format, settings["columns"])
    if rows != manifest["rows"] or (settings["limit"] is not None and rows > settings["limit"]):
        raise ValueError("Extracted row count differs")
    if manifest["source_audit"] and rows > manifest["source_audit"]["rows"]:
        raise ValueError("Extraction contains more rows than the audited input")
    if source is not None and file_sha256(source) != manifest["input_sha256"]:
        raise ValueError("Intermediate file checksum differs from the extraction")
    return manifest
