"""Read a classification CSV row by row and profile its contents without fitting a model."""

from collections import Counter
from contextlib import closing
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import re
import sqlite3
from tempfile import TemporaryDirectory


MISSING = {
    "", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan",
    "1.#IND", "1.#QNAN", "<NA>", "N/A", "NA", "NULL", "NaN", "None", "n/a", "nan", "null",
}
NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")
NONFINITE = {"inf", "infinity", "nan"}
CLASS_LIMIT = 100
EXAMPLE_LIMIT = 10
COLUMN_ARGUMENTS = (("target", "target"), ("id_column", "id"), ("time_column", "time"))


@dataclass
class ColumnProfile:
    name: str
    role: str
    track_decimal: bool = False
    counts: Counter = field(default_factory=Counter)
    minimum: float | None = None
    maximum: float | None = None
    decimal_shape: tuple[int, int] = (0, 0)

    def observe(self, value):
        token = value.strip()
        if token in MISSING:
            kind = "missing"
        elif token.lower().lstrip("+-") in NONFINITE:
            kind = "nonfinite"
        elif NUMBER.fullmatch(token):
            number = float(token)
            kind = "numeric" if isfinite(number) else "nonfinite"
            if kind == "numeric":
                self.minimum = number if self.minimum is None else min(self.minimum, number)
                self.maximum = number if self.maximum is None else max(self.maximum, number)
                if self.track_decimal and sum(self.decimal_shape) <= 38:
                    self.observe_decimal(token)
        elif token.lower() in ("true", "false"):
            kind = "boolean"
        else:
            kind = "text"
        self.counts[kind] += 1

    def observe_decimal(self, token):
        if re.match(r"[+-]?0[0-9]", token):
            self.decimal_shape = (39, 0)
            return
        try:
            _, digits, exponent = Decimal(token).as_tuple()
        except InvalidOperation:
            self.decimal_shape = (39, 0)
            return
        integer_digits, scale = self.decimal_shape
        self.decimal_shape = (max(integer_digits, len(digits) + exponent), max(scale, -exponent))

    def export(self, position):
        kinds = [kind for kind in ("numeric", "boolean", "text") if self.counts[kind]]
        observed = "mixed" if len(kinds) > 1 else next(iter(kinds), "undetermined")
        result = {
            "position": position, "name": self.name, "role": self.role, "observed_type": observed,
            **{kind: self.counts[kind] for kind in ("missing", "nonfinite", "numeric", "boolean", "text")},
            "numeric_min": self.minimum, "numeric_max": self.maximum,
        }
        if self.track_decimal:
            result["decimal_shape"] = list(self.decimal_shape)
        return result


@dataclass
class TimeProfile:
    parsed: int = 0
    invalid: int = 0
    ranges: dict = field(default_factory=dict)

    def observe(self, value):
        if value.strip() in MISSING:
            return
        try:
            point = datetime.fromisoformat(value.strip())
            zone = "utc" if point.tzinfo is not None else "local_unspecified"
            if zone == "utc":
                point = point.astimezone(timezone.utc)
        except (ValueError, OverflowError):
            self.invalid += 1
            return
        self.parsed += 1
        start, end = self.ranges.get(zone, (point, point))
        self.ranges[zone] = (min(start, point), max(end, point))

    def export(self, status, missing):
        clock = "mixed_timezones" if len(self.ranges) > 1 else next(iter(self.ranges), "undetermined")
        bounds = self.ranges.get(clock)
        return {
            "status": status, "parsed": self.parsed if status == "counted" else None,
            "invalid": self.invalid if status == "counted" else None, "missing": missing,
            "clock": clock, "earliest": bounds[0].isoformat() if bounds else None,
            "latest": bounds[1].isoformat() if bounds else None,
        }


def audit_settings(csv_path, target=None, id_column=None, time_column=None, prediction_goal=None,
                   prediction_time=None, row_unit=None, label_source=None):
    values = {
        "target": target, "id_column": id_column, "time_column": time_column,
        "prediction_goal": prediction_goal, "prediction_time": prediction_time, "row_unit": row_unit,
        "label_source": label_source,
    }
    for name, value in values.items():
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{name} must be a non-empty string")
    declared = [values[name] for name, _ in COLUMN_ARGUMENTS if values[name] is not None]
    if len(declared) != len(set(declared)):
        raise ValueError("Target, identifier and timestamp must refer to different columns")
    return {"csv": str(csv_path), **values}


def resolve_roles(header, settings):
    counts = Counter(header)
    positions, statuses = {}, {}
    for name, role in COLUMN_ARGUMENTS:
        value = settings[name]
        if value is None:
            statuses[role] = "not_declared"
        elif counts[value] == 1:
            positions[role] = header.index(value)
            statuses[role] = "counted"
        else:
            statuses[role] = "ambiguous" if counts[value] > 1 else "not_found"
    return counts, positions, statuses


def record_key(values):
    raw = json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return sha256(raw).digest()


def initialize(db):
    db.executescript("""
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA cache_size=-4096;
        PRAGMA temp_store=FILE;
        CREATE TABLE rows (key BLOB PRIMARY KEY, n INTEGER, line INTEGER) WITHOUT ROWID;
        CREATE TABLE features (key BLOB PRIMARY KEY, n INTEGER, line INTEGER, label TEXT,
                               conflict INTEGER) WITHOUT ROWID;
        CREATE TABLE labels (label TEXT PRIMARY KEY, n INTEGER) WITHOUT ROWID;
        CREATE TABLE ids (key BLOB PRIMARY KEY, n INTEGER) WITHOUT ROWID;
    """)


def record_row(db, row, line, label, identifier, feature_positions):
    db.execute("INSERT INTO rows VALUES (?,1,?) ON CONFLICT(key) DO UPDATE SET n=n+1", (record_key(row), line))
    if label is not None:
        db.execute("INSERT INTO labels VALUES (?,1) ON CONFLICT(label) DO UPDATE SET n=n+1", (label,))
    if identifier is not None:
        key = sha256(identifier.encode("utf-8")).digest()
        db.execute("INSERT INTO ids VALUES (?,1) ON CONFLICT(key) DO UPDATE SET n=n+1", (key,))
    if feature_positions:
        key = record_key([row[position] for position in feature_positions])
        db.execute("""
            INSERT INTO features VALUES (?,1,?,?,0)
            ON CONFLICT(key) DO UPDATE SET n=n+1,
                conflict=CASE WHEN label != excluded.label THEN 1 ELSE conflict END,
                label=COALESCE(label,excluded.label)
        """, (key, line, label))


def duplicates(db, table):
    groups, extra_rows = db.execute(f"SELECT COUNT(*), COALESCE(SUM(n-1),0) FROM {table} WHERE n>1").fetchone()
    examples = db.execute(
        f"SELECT line,n FROM {table} WHERE n>1 ORDER BY line LIMIT ?", (EXAMPLE_LIMIT,),
    )
    return {
        "groups": groups, "extra_rows": extra_rows,
        "examples": [{"first_line": line, "rows": count} for line, count in examples],
    }


def class_profile(db, status, missing):
    if status != "counted":
        return {
            "status": status, "distinct": None, "missing": None, "classes_under_six": None,
            "other_classes": None, "other_rows": None, "classes": [],
        }
    count, labeled, low = db.execute(
        "SELECT COUNT(*), COALESCE(SUM(n),0), COUNT(CASE WHEN n<6 THEN 1 END) FROM labels",
    ).fetchone()
    classes = [{"label": label, "rows": n} for label, n in db.execute(
        "SELECT label,n FROM labels ORDER BY n DESC,label LIMIT ?", (CLASS_LIMIT,),
    )]
    return {
        "status": status, "distinct": count, "missing": missing, "classes_under_six": low,
        "other_classes": count - len(classes), "other_rows": labeled - sum(item["rows"] for item in classes),
        "classes": classes,
    }


def id_profile(db, status, missing):
    if status != "counted":
        return {"status": status, "missing": None, "distinct": None, "repeated_ids": None, "rows_with_repeated_ids": None}
    distinct, repeated, rows = db.execute(
        "SELECT COUNT(*), COUNT(CASE WHEN n>1 THEN 1 END), COALESCE(SUM(CASE WHEN n>1 THEN n ELSE 0 END),0) FROM ids"
    ).fetchone()
    return {"status": status, "missing": missing, "distinct": distinct, "repeated_ids": repeated, "rows_with_repeated_ids": rows}


def read_record(reader):
    start_line = reader.line_num + 1
    try:
        return next(reader, None), start_line, None
    except (csv.Error, UnicodeDecodeError) as error:
        return None, start_line, {"line": start_line, "type": type(error).__name__}


def scan_records(reader, db, header, profiles, positions):
    scan = {
        "status": "complete", "records": 0, "valid_rows": 0, "blank_records": 0,
        "wrong_width_rows": 0, "wrong_width_examples": [], "parse_error": None,
    }
    clock = TimeProfile()
    time_position = positions.get("time")
    feature_positions = [i for i in range(len(header)) if i not in positions.values()] if "target" in positions else []

    def nonmissing(row, role):
        if role in positions:
            value = row[positions[role]]
            return value if value.strip() not in MISSING else None
        return None

    while header:
        row, line, error = read_record(reader)
        if row is None:
            if error:
                scan.update(status="partial", parse_error=error)
            break
        if not row:
            scan["blank_records"] += 1
            continue
        scan["records"] += 1
        if len(row) != len(header):
            scan["wrong_width_rows"] += 1
            if len(scan["wrong_width_examples"]) < EXAMPLE_LIMIT:
                scan["wrong_width_examples"].append({"line": line, "fields": len(row)})
            continue
        scan["valid_rows"] += 1
        for profile, value in zip(profiles, row):
            profile.observe(value)
        label, identifier = (nonmissing(row, role) for role in ("target", "id"))
        record_row(db, row, line, label, identifier, feature_positions)
        if time_position is not None:
            clock.observe(row[time_position])
    return scan, clock, bool(feature_positions)


def scan_csv(csv_path, settings):
    with (TemporaryDirectory(prefix="diy-data-audit-") as tmp,
          closing(sqlite3.connect(Path(tmp) / "profile.sqlite")) as db,
          Path(csv_path).open(encoding="utf-8-sig", newline="") as stream):
        initialize(db)
        reader = csv.reader(stream, strict=True)
        header, _, error = read_record(reader)
        leading_blanks = 0
        while header == []:
            leading_blanks += 1
            header, _, error = read_record(reader)
        header = header or []
        counts, positions, statuses = resolve_roles(header, settings)
        roles = {position: role for role, position in positions.items()}
        numeric_types = settings.get("conversion_types") == "numeric"
        profiles = [
            ColumnProfile(name, roles.get(i, "feature"), track_decimal=numeric_types and i not in roles)
            for i, name in enumerate(header)
        ]
        scan, clock, has_feature_groups = scan_records(reader, db, header, profiles, positions)
        if not header:
            scan.update(status="partial" if error else "no_header", parse_error=error)
        scan["blank_records"] += leading_blanks
        scan["columns"] = len(header)

        def missing(role):
            return profiles[positions[role]].counts["missing"] if role in positions else None

        features = duplicates(db, "features") if has_feature_groups else None
        if features is not None:
            features["conflicting_label_groups"] = db.execute("SELECT COUNT(*) FROM features WHERE conflict=1").fetchone()[0]
        return {
            "scan": scan, "columns": [column.export(i) for i, column in enumerate(profiles, start=1)],
            "duplicate_header_names": [name for name, count in counts.items() if count > 1],
            "empty_header_columns": [i for i, name in enumerate(header, start=1) if not name.strip()],
            "duplicates": {"whole_rows": duplicates(db, "rows"), "features": features},
            "target": class_profile(db, statuses["target"], missing("target")),
            "id": id_profile(db, statuses["id"], missing("id")),
            "time": clock.export(statuses["time"], missing("time")),
        }
