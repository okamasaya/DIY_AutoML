"""Save a pre-training CSV audit and the evaluation context supplied by the caller."""

from pathlib import Path
import re

from openpyxl import Workbook
from openpyxl.cell import WriteOnlyCell
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter

from .data_conversion import CONVERSION_SCHEMA, conversion_settings, convert_csv
from .data_profile import COLUMN_ARGUMENTS, audit_settings, scan_csv
from .reporting import fail, file_sha256, prepare, write_json


SCHEMA = 15
METADATA = {
    "prediction_goal": "Prediction goal", "prediction_time": "At what point to predict",
    "row_unit": "What one row represents", "label_source": "Label source",
    "target": "Target column", "id_column": "Identifier column", "time_column": "Timestamp column",
}
SHEET_WIDTHS = {
    "Summary": (28, 70), "Review": (33, 13, 94),
    "Columns": (12, 30, 14, 19), "Class_Counts": (32, 15, 14),
    "Duplicates": (18, 20, 14, 27), "Examples": (19, 18, 16, 18),
    "Evaluation_Context": (36, 74), "Conversion": (40, 90),
    "まず読む": (34, 90),
}


def audit_issues(profile, settings):
    scan, columns = profile["scan"], profile["columns"]
    target, identifier, clock = (profile[key] for key in ("target", "id", "time"))
    issues = []

    def check(count, code, action):
        if count:
            issues.append({"code": code, "count": int(count), "next_check": action})

    check(scan["status"] != "complete", "incomplete_input", "Check the header, UTF-8 encoding or CSV quoting.")
    check(scan["valid_rows"] == 0, "no_valid_rows", "Supply data records with the same number of cells as the header.")
    check(scan["wrong_width_rows"], "row_width", "Check the listed lines and the CSV separator.")
    check(len(profile["duplicate_header_names"]), "duplicate_header", "Give each column a distinct name.")
    check(len(profile["empty_header_columns"]), "empty_header", "Name the listed columns.")
    for key, role in COLUMN_ARGUMENTS:
        status = profile[role]["status"]
        if status in ("not_found", "ambiguous"):
            check(1, f"{role}_{status}", f"Confirm the column name supplied as {key}.")
    check(target["status"] == "not_declared", "target_not_declared", "Specify the target column to count labels.")
    if target["status"] == "counted":
        check(target["distinct"] < 2, "fewer_than_two_classes", "Confirm the class definitions and collect labeled examples.")
        check(target["classes_under_six"], "classes_under_six_rows",
              "At least five training rows and one test row per class are needed for this evaluation plan.")
        check(target["missing"], "missing_target", "Check the origin of the missing labels.")
    features = [column for column in columns if column["role"] == "feature"]
    check(not features and bool(columns), "no_feature_columns", "Identify which columns are available as predictors.")
    check(sum(column["missing"] for column in features), "missing_feature_values", "Check blank cells and the meaning of missing values.")
    check(sum(column["nonfinite"] for column in features), "nonfinite_feature_values", "Confirm infinite and out-of-range numeric values.")
    check(sum(column["observed_type"] in ("mixed", "text") for column in features),
          "nonnumeric_feature_columns", "Inspect text and mixed columns before using the numeric CSV workflow.")
    check(profile["duplicates"]["whole_rows"]["extra_rows"], "duplicate_rows", "Check whether identical records repeat the same observation.")
    feature_groups = profile["duplicates"]["features"]
    if feature_groups:
        check(feature_groups["extra_rows"], "duplicate_features", "Check equal predictor rows and whether their evaluation units overlap.")
        check(feature_groups["conflicting_label_groups"], "conflicting_labels", "Compare the labels with the original records for these groups.")
    check(identifier["repeated_ids"], "repeated_ids", "Consider keeping each identifier together when splitting.")
    check(identifier["missing"], "missing_identifier", "Check which rows can be assigned to a group.")
    check(clock["invalid"], "unparsed_timestamp", "Check the timestamp format and units.")
    check(clock["missing"], "missing_timestamp", "Check which rows can be placed in time.")
    check(clock["clock"] == "mixed_timezones", "mixed_timezones", "Resolve the time zones before comparing dates.")
    check(sum(settings[key] is not None for key in ("id_column", "time_column")),
          "metadata_columns", "Review the input feature list. The current training commands take every non-target column.")
    return issues


def build_profile(csv_path, settings, input_hash):
    profile = scan_csv(csv_path, settings)
    if settings.get("bp_summary"):
        profile["source_bytes"] = Path(csv_path).stat().st_size
    if file_sha256(csv_path) != input_hash:
        raise ValueError("The input CSV changed during the audit")
    return {
        "input_sha256": input_hash, "settings": settings, **profile,
        "issues": audit_issues(profile, settings),
    }


def bp_summary_rows(profile):
    scan, target = profile["scan"], profile["target"]
    rows = [
        ("確認する項目", "結果"),
        ("読み取った範囲", "末尾まで" if scan["status"] == "complete" else "入力の確認が必要"),
        ("列数が合ったデータ行", scan["valid_rows"]),
        ("列数が合わない行", scan["wrong_width_rows"]),
        ("欠損セル数", sum(column["missing"] for column in profile["columns"])),
        ("全列が一致する重複行", profile["duplicates"]["whole_rows"]["extra_rows"]),
        ("正解ラベルの種類数", target["distinct"] if target["status"] == "counted" else "未確認"),
        ("確認項目数", len(profile["issues"])),
        ("元CSVのサイズ（バイト）", profile["source_bytes"]),
    ]
    conversion = profile.get("conversion")
    if conversion is None:
        rows.append(("作業用ファイルを圧縮", "指定なし"))
    else:
        status = {"completed": "保存済み", "blocked": "元データの確認が必要", "failed": "圧縮処理の確認が必要"}
        rows.append(("作業用ファイルを圧縮", status[conversion["status"]]))
        file = conversion["files"].get("parquet")
        if file:
            rows.extend([
                ("圧縮した作業用ファイル", file["path"]),
                ("圧縮後のサイズ（バイト）", file["bytes"]),
                ("圧縮後のサイズ（元CSV比 %）", round(100 * file["bytes"] / profile["source_bytes"], 1)),
            ])
        else:
            rows.append(("圧縮の詳細", "Conversionシートを確認してください"))
    rows.append(("点検の詳細", "ReviewとColumnsシートを確認してください"))
    return rows


def report_rows(profile):
    scan, target, identifier, clock = (profile[key] for key in ("scan", "target", "id", "time"))
    settings = profile["settings"]
    duplicates = profile["duplicates"]
    columns = ("position", "name", "role", "observed_type", "missing", "nonfinite", "numeric", "boolean", "text",
               "numeric_min", "numeric_max")
    summary = [
        ("Item", "Value"), ("Input SHA-256", profile["input_sha256"]), ("Scan status", scan["status"]),
        ("Records read", scan["records"]), ("Rows with expected width", scan["valid_rows"]),
        ("Blank records", scan["blank_records"]), ("Wrong-width records", scan["wrong_width_rows"]),
        ("Columns", scan["columns"]), ("Duplicate header names", len(profile["duplicate_header_names"])),
        ("Empty header cells", len(profile["empty_header_columns"])), ("Items to review", len(profile["issues"])),
    ]
    if scan["parse_error"]:
        summary.extend((("Parse stopped near line", scan["parse_error"]["line"]),
                        ("Parse error type", scan["parse_error"]["type"])))
    sheets = {
        "Summary": summary,
        "Review": [("Check", "Count", "What to review"), *(
            (issue["code"], issue["count"], issue["next_check"]) for issue in profile["issues"]
        )],
        "Columns": [tuple(key.replace("_", " ").title() for key in columns), *(
            tuple(row[key] for key in columns) for row in profile["columns"]
        )],
        "Class_Counts": [("Class", "Rows", "Kind"),
                         *((row["label"], row["rows"], "observed") for row in target["classes"]),
                         ("Missing target", target["missing"], "summary"),
                         ("Other classes", target["other_rows"], "summary")],
        "Duplicates": [("Scope", "Repeated groups", "Extra rows", "Conflicting label groups"),
                       *((name, row["groups"], row["extra_rows"], row.get("conflicting_label_groups"))
                         for name, row in duplicates.items() if row is not None)],
        "Examples": [("Scope", "First CSV line", "Rows in group", "Observed fields"),
                     *((name, example["first_line"], example["rows"], None)
                       for name, row in duplicates.items() if row is not None for example in row["examples"]),
                     *(("wrong_width", example["line"], None, example["fields"])
                       for example in scan["wrong_width_examples"])],
        "Evaluation_Context": [
            ("Item", "Value"),
            *((title, settings[name] if settings[name] is not None else "unknown") for name, title in METADATA.items()),
            ("Target status", target["status"]), ("Distinct observed classes", target["distinct"]),
            ("Classes with fewer than six rows", target["classes_under_six"]),
            ("Classes omitted from table", target["other_classes"]),
            ("Identifier status", identifier["status"]),
            ("Distinct identifiers", identifier["distinct"] if identifier["status"] == "counted" else "unknown"),
            ("Repeated identifiers", identifier["repeated_ids"] if identifier["status"] == "counted" else "unknown"),
            ("Rows sharing an identifier", identifier["rows_with_repeated_ids"]),
            ("Missing identifiers", identifier["missing"]), ("Timestamp status", clock["status"]),
            ("Time zone", clock["clock"]), ("Parsed timestamps", clock["parsed"]),
            ("Unparsed timestamps", clock["invalid"]), ("Missing timestamps", clock["missing"]),
            ("Earliest", clock["earliest"]), ("Latest", clock["latest"]),
        ],
    }
    if "conversion" in profile:
        conversion = profile["conversion"]
        sheets["Conversion"] = [
            ("Item", "Value"), ("Requested formats", ", ".join(settings["convert"])),
            ("Type mode", settings["conversion_types"]), ("Status", conversion["status"]),
            ("Details", conversion["reason"]), ("Rows exported", conversion.get("rows")),
            *((kind, info["path"]) for kind, info in conversion["files"].items()),
            *((f"Column - {column['name']}", column["type"]) for column in conversion.get("columns", ())),
        ]
    if settings.get("bp_summary"):
        return {"まず読む": bp_summary_rows(profile), **sheets}
    return sheets


def excel_row(row):
    return tuple(
        excel_value(value)
        for value in row
    )


def excel_value(value):
    if isinstance(value, str):
        return re.sub(r"[\ufffe\uffff]", "", ILLEGAL_CHARACTERS_RE.sub("", value))[:32767] or None
    if isinstance(value, float) and abs(value) >= 1e300:
        return repr(value)
    return value


def write_workbook(path, sheets):
    book = Workbook(write_only=True)
    heading = Font(bold=True)
    for name, rows in sheets.items():
        sheet = book.create_sheet(name)
        sheet.freeze_panes = "A2"
        for index, width in enumerate(SHEET_WIDTHS[name], start=1):
            sheet.column_dimensions[get_column_letter(index)].width = width
        for index, raw in enumerate(rows):
            row = []
            for value in excel_row(raw):
                if isinstance(value, str):
                    cell = WriteOnlyCell(sheet, value=value)
                    cell.data_type = "s"
                    if index == 0:
                        cell.font = heading
                    row.append(cell)
                else:
                    row.append(value)
            sheet.append(row)
    book.save(path)


def audit_data_csv(csv, output, target=None, id_column=None, time_column=None, prediction_goal=None,
                   prediction_time=None, row_unit=None, label_source=None, convert=(), conversion_types="preserve",
                   bp_summary=False):
    """Inspect a source CSV and optionally export its records in portable binary formats."""
    settings = audit_settings(csv, target, id_column, time_column, prediction_goal, prediction_time, row_unit, label_source)
    conversion = conversion_settings(convert, conversion_types)
    settings.update(conversion)
    if bp_summary:
        settings["bp_summary"] = True
    output, manifest = prepare(output, CONVERSION_SCHEMA if conversion else SCHEMA, settings)
    try:
        profile = build_profile(Path(csv), settings, manifest["input_sha256"])
        converted = {}
        if conversion:
            profile["conversion"] = convert_csv(Path(csv), output, profile)
            manifest["conversion"] = profile["conversion"]
            converted = {info["path"]: info["sha256"] for info in profile["conversion"]["files"].values()}
        sheets = report_rows(profile)
        write_workbook(output / "report.xlsx", sheets)
        write_json(output / "profile.json", profile)
        manifest.update(
            status="completed", scan_status=profile["scan"]["status"],
            sheets={name: len(rows) - 1 for name, rows in sheets.items()},
            artifacts={**{name: file_sha256(output / name) for name in ("profile.json", "report.xlsx")}, **converted},
        )
        write_json(output / "manifest.json", manifest)
        return output
    except Exception as error:
        fail(output, manifest, error)
        raise


def prepare_bp_csv(csv, output, target=None, compress=False):
    """Write a Japanese data audit and optionally archive a compressed working copy."""
    if not isinstance(compress, bool):
        raise ValueError("compress must be True or False")
    formats = ("parquet",) if compress else ()
    return audit_data_csv(csv, output, target=target, convert=formats, bp_summary=True)
