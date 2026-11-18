"""Check an extracted CSV and optionally confirm its input snapshot."""

import csv
import json
from pathlib import Path

from .data_extract import CSV_ARTIFACT, SCHEMA, extraction_settings
from .reporting import file_sha256


def validate_data_extract(directory, source=None):
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != SCHEMA or manifest["status"] != "completed":
        raise ValueError("Data extraction did not complete")
    settings = manifest["settings"]
    if settings != extraction_settings(**settings):
        raise ValueError("Extraction settings differ")
    if set(manifest["artifacts"]) != {CSV_ARTIFACT}:
        raise ValueError("Unexpected extraction artifact set")
    data = root / CSV_ARTIFACT
    if file_sha256(data) != manifest["artifacts"][CSV_ARTIFACT]:
        raise ValueError("Extracted CSV checksum differs")
    source_columns = {column["name"] for column in manifest["source_columns"]}
    requested = set(settings["columns"]) | {term[0] for term in settings["where"]}
    if not requested <= source_columns:
        raise ValueError("Selected or filtered column is absent from source schema")
    if manifest["source_audit"] and manifest["source_audit"]["columns"] != manifest["source_columns"]:
        raise ValueError("Recorded columns differ from the source audit")
    with data.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        if next(reader, None) != settings["columns"]:
            raise ValueError("Extracted CSV header differs")
        rows = 0
        for row in reader:
            if len(row) != len(settings["columns"]):
                raise ValueError("Extracted CSV record width differs")
            rows += 1
    if rows != manifest["rows"] or (settings["limit"] is not None and rows > settings["limit"]):
        raise ValueError("Extracted CSV row count differs")
    if manifest["source_audit"] and rows > manifest["source_audit"]["rows"]:
        raise ValueError("Extracted CSV contains more rows than the audited input")
    if source is not None and file_sha256(source) != manifest["input_sha256"]:
        raise ValueError("Intermediate file checksum differs from the extraction")
    return manifest
