"""Check a saved CSV audit and optionally reproduce it from its source file."""

from itertools import zip_longest
import json
from math import isclose
from pathlib import Path

from openpyxl import load_workbook

from .data_audit import SCHEMA, audit_issues, build_profile, excel_row, report_rows
from .data_conversion import CONVERSION_SCHEMA, conversion_blocker, conversion_settings, converted_path, storage_columns, verify_formats
from .data_profile import audit_settings
from .reporting import file_sha256


def same_row(expected, actual):
    if actual is None or expected is None or any(cell.data_type == "f" for cell in actual):
        return False
    wanted = excel_row(expected)
    seen = tuple(cell.value for cell in actual)
    return all(
        isinstance(b, (int, float)) and isclose(a, b, rel_tol=1e-12, abs_tol=0)
        if isinstance(a, float) else a == b
        for a, b in zip_longest(wanted, seen)
    )


def check_conversion(root, profile, manifest):
    request = profile["settings"]["convert"]
    conversion = profile["conversion"]
    if conversion != manifest["conversion"]:
        raise ValueError("Conversion record differs")
    status = conversion["status"]
    blocker = conversion_blocker(profile, request)
    if status == "completed":
        if blocker or conversion["reason"] is not None or set(conversion["files"]) != set(request):
            raise ValueError("Unexpected set of converted files")
        if conversion["rows"] != profile["scan"]["valid_rows"] or conversion["columns"] != storage_columns(profile):
            raise ValueError("Conversion row count or expected columns differ")
        expected_paths = {name: converted_path(name) for name in request}
        for name, info in conversion["files"].items():
            if info["path"] != expected_paths[name] or manifest["artifacts"].get(info["path"]) != info["sha256"]:
                raise ValueError("Converted file path or checksum differs")
        return expected_paths
    if conversion["files"] or (root / "converted").exists():
        raise ValueError("An unsuccessful conversion contains published output")
    if status == "blocked" and blocker and conversion["reason"] == blocker:
        return {}
    if status == "failed" and not blocker and conversion["reason"]:
        return {}
    raise ValueError("Unexpected conversion status")


def validate_data_audit(directory, csv=None):
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    schema = manifest["schema_version"]
    if schema not in (SCHEMA, CONVERSION_SCHEMA) or manifest["status"] != "completed":
        raise ValueError("Data audit did not finish writing a report")

    profile = json.loads((root / "profile.json").read_text(encoding="utf-8"))
    settings = profile["settings"]
    base = audit_settings(settings["csv"], **{
        key: value for key, value in settings.items() if key not in ("csv", "convert", "conversion_types", "bp_summary")
    })
    if settings.get("bp_summary") is True:
        base["bp_summary"] = True
    paths = {}
    if schema == CONVERSION_SCHEMA:
        request = conversion_settings(settings["convert"], settings["conversion_types"])
        if not request:
            raise ValueError("No conversion was requested")
        base.update(request)
        paths = check_conversion(root, profile, manifest)
    elif "conversion" in profile or "conversion" in manifest:
        raise ValueError("Unexpected conversion record")
    if settings != manifest["settings"] or settings != base:
        raise ValueError("Audit settings differ")
    if set(manifest["artifacts"]) != {"profile.json", "report.xlsx", *paths.values()}:
        raise ValueError("Unexpected data audit artifact set")
    for name, digest in manifest["artifacts"].items():
        if file_sha256(root / name) != digest:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    if profile["input_sha256"] != manifest["input_sha256"] or profile["scan"]["status"] != manifest["scan_status"]:
        raise ValueError("Audit source or scan status differs")
    if profile["issues"] != audit_issues(profile, settings):
        raise ValueError("Audit review items differ")
    sheets = report_rows(profile)
    if manifest["sheets"] != {name: len(rows) - 1 for name, rows in sheets.items()}:
        raise ValueError("Audit sheet row counts differ")
    workbook = load_workbook(root / "report.xlsx", read_only=True)
    try:
        if workbook.sheetnames != list(sheets):
            raise ValueError("Audit worksheet names differ")
        for name, expected in sheets.items():
            for row, cells in zip_longest(expected, workbook[name].iter_rows()):
                if not same_row(row, cells):
                    raise ValueError(f"Audit worksheet differs: {name}")
    finally:
        workbook.close()

    if paths:
        files = {name: root / path for name, path in paths.items()}
        conversion = profile["conversion"]
        for name, path in files.items():
            if path.stat().st_size != conversion["files"][name]["bytes"]:
                raise ValueError(f"Converted file size differs: {name}")
        verify_formats(files, conversion["columns"], profile)
    if csv is not None:
        digest = file_sha256(csv)
        if digest != manifest["input_sha256"]:
            raise ValueError("Input CSV checksum differs from the saved audit")
        if {key: value for key, value in profile.items() if key != "conversion"} != build_profile(csv, settings, digest):
            raise ValueError("Recalculated CSV profile differs")
    return manifest
