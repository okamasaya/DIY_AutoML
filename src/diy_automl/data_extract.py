"""Export selected columns from CSV or from a portable storage file."""

from collections import Counter
from contextlib import closing
import csv
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
from math import isfinite
from numbers import Integral
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from .data_conversion import CONVERSION_SCHEMA, DUCKDB_CONFIG, FORMATS, converted_path, sql_identifier
from .data_profile import NUMBER, read_header
from .reporting import fail, file_sha256, prepare, write_json


SCHEMA = 17
OPERATORS = {"eq": "=", "ne": "<>", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}
INTEGER_BITS = {"TINYINT": 8, "SMALLINT": 16, "INTEGER": 32, "BIGINT": 64, "HUGEINT": 128}
DECIMAL_TYPE = re.compile(r"DECIMAL\(([0-9]+),([0-9]+)\)")
CSV_ARTIFACT = "data.csv"


def extraction_settings(source, columns, where=(), limit=None):
    kind = Path(source).suffix.lower()[1:]
    if kind not in ("csv", *FORMATS):
        raise ValueError("Source must be a .csv, .parquet or .duckdb file")
    selected = [columns] if isinstance(columns, str) else list(columns)
    if not selected or any(not isinstance(name, str) or not name.strip() for name in selected):
        raise ValueError("Select at least one named output column")
    if len(selected) != len(set(selected)):
        raise ValueError("Select each output column only once")
    filters = []
    for term in where:
        if not isinstance(term, (tuple, list)) or len(term) != 3:
            raise ValueError("Each filter requires a column, operator and value")
        name, operator, value = term
        if not isinstance(name, str) or not name.strip() or not isinstance(operator, str) or operator not in OPERATORS:
            raise ValueError("Filter operators are eq, ne, lt, le, gt and ge")
        if not isinstance(value, (str, int, float, Decimal)):
            raise ValueError("Filter values must be scalar strings, numbers or booleans")
        filters.append([name, operator, str(value)])
    if kind == "csv" and filters:
        raise ValueError("Row filters require a Parquet or DuckDB source")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, Integral) or not 0 <= limit < 2**63):
        raise ValueError("limit must be a nonnegative 64-bit integer")
    return {"source": str(source), "columns": selected, "where": filters, "limit": int(limit) if limit is not None else None}


def exact_numeric(value, dtype, precision, scale):
    token = value.strip()
    try:
        if not NUMBER.fullmatch(token):
            raise InvalidOperation
        number = Decimal(token)
    except InvalidOperation as error:
        raise ValueError(f"Use a finite number for a {dtype} filter") from error
    if number.is_zero():
        return "0"
    _, digits, exponent = number.as_tuple()
    size = len(digits)
    while digits[size - 1] == 0:
        size -= 1
        exponent += 1
    if max(0, size + exponent) > precision - scale or max(0, -exponent) > scale:
        raise ValueError(f"Value {value!r} does not fit {dtype} exactly")
    return format(number, "f")


def bound_filter(dtype, operator, value, parameter):
    if dtype == "VARCHAR":
        if operator not in ("eq", "ne"):
            raise ValueError("Text filters use eq or ne. For numeric ranges, convert the feature to a numeric type first.")
        return parameter, value
    if dtype == "BOOLEAN":
        if operator not in ("eq", "ne") or value.lower() not in ("true", "false"):
            raise ValueError("Boolean filters use eq or ne with true or false")
        return parameter, value.lower() == "true"
    decimal = DECIMAL_TYPE.fullmatch(dtype)
    if decimal:
        precision, scale = map(int, decimal.groups())
        return f"CAST({parameter} AS {dtype})", exact_numeric(value, dtype, precision, scale)
    unsigned = dtype.startswith("U")
    bits = INTEGER_BITS.get(dtype[1:] if unsigned else dtype)
    if bits:
        number = int(Decimal(exact_numeric(value, dtype, 39, 0)))
        bound = 2**(bits if unsigned else bits - 1)
        if not (0 if unsigned else -bound) <= number < bound:
            raise ValueError(f"Value {value!r} does not fit {dtype}")
        return f"CAST({parameter} AS {dtype})", str(number)
    if dtype in ("FLOAT", "DOUBLE"):
        if not NUMBER.fullmatch(value.strip()) or not isfinite(float(value)):
            raise ValueError(f"Use a finite number for a {dtype} filter")
        if dtype == "FLOAT" and abs(float(value)) > float.fromhex("0x1.fffffep127"):
            raise ValueError(f"Value {value!r} does not fit {dtype}")
        return f"CAST({parameter} AS {dtype})", value
    raise ValueError(f"Filters for {dtype} columns are unsupported")


def audit_lineage(source, digest, storage):
    audit = source.parent.parent / "manifest.json"
    if source.parent.name != "converted" or not audit.is_file():
        return None
    raw = audit.read_bytes()
    record = json.loads(raw)
    if record.get("schema_version") != CONVERSION_SCHEMA:
        return None
    conversion = record["conversion"]
    name = converted_path(storage)
    file = conversion["files"].get(storage)
    if (record["status"] != "completed" or conversion["status"] != "completed" or not file
            or file["path"] != name or source.name != Path(name).name
            or file["sha256"] != digest or record["artifacts"].get(name) != digest):
        raise ValueError("Source file does not match its conversion manifest")
    return {
        "audit_manifest_sha256": sha256(raw).hexdigest(), "original_csv_sha256": record["input_sha256"],
        "columns": conversion["columns"], "rows": conversion["rows"],
    }


def projection(relation, inputs, settings, schema):
    selected = settings["columns"]
    needed = set(selected) | {name for name, _, _ in settings["where"]}
    if absent := sorted(needed - schema.keys()):
        raise ValueError(f"Columns not found in source: {', '.join(absent)}")
    fields = ", ".join(map(sql_identifier, selected))
    conditions, parameters = [], dict(inputs)
    for index, (name, operator, value) in enumerate(settings["where"]):
        key = f"filter_{index}"
        operand, parameters[key] = bound_filter(schema[name], operator, value, f"${key}")
        conditions.append(f"{sql_identifier(name)} {OPERATORS[operator]} {operand}")
    query = f"SELECT {fields} FROM {relation}"
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    if settings["limit"] is not None:
        query += " LIMIT $row_limit"
        parameters["row_limit"] = settings["limit"]
    return query, parameters


def export_csv(source, destination, scratch, settings, lineage):
    try:
        import duckdb
    except ImportError as error:
        raise RuntimeError("Install the optional storage dependencies with pip install -e '.[storage]'") from error

    is_parquet = source.suffix.lower() == ".parquet"
    config = {
        **DUCKDB_CONFIG, "temp_directory": (scratch / "spill").as_posix(), "preserve_insertion_order": "true",
        "autoload_known_extensions": "false", "autoinstall_known_extensions": "false",
    }
    with closing(duckdb.connect(":memory:" if is_parquet else str(source), read_only=not is_parquet, config=config)) as db:
        paths = [destination.as_posix()]
        if is_parquet:
            paths.append(source.as_posix())
        db.execute("SET allowed_paths = ?", [paths])
        db.execute("SET enable_external_access = false")

        if is_parquet:
            relation, inputs = "read_parquet($source, hive_partitioning=false)", {"source": source.as_posix()}
        else:
            table = db.execute("""
                SELECT table_type FROM information_schema.tables
                WHERE table_catalog = current_database() AND table_schema = 'main' AND table_name = 'data'
            """).fetchone()
            if table != ("BASE TABLE",):
                raise ValueError("DuckDB source must contain the base table main.data")
            database = db.execute("SELECT current_database()").fetchone()[0]
            relation, inputs = f"{sql_identifier(database)}.main.data", {}
        description = db.execute(f"DESCRIBE SELECT * FROM {relation}", inputs).fetchall()
        schema = {row[0]: row[1] for row in description}
        columns = [{"name": name, "type": dtype} for name, dtype in schema.items()]
        if lineage and columns != lineage["columns"]:
            raise ValueError("SQL column names or types differ from the audited conversion")
        query, parameters = projection(relation, inputs, settings, schema)
        parameters["file_out"] = destination.as_posix()
        rows = db.execute(f"COPY ({query}) TO $file_out (FORMAT CSV, HEADER true)", parameters).fetchone()[0]
    return rows, columns, duckdb.__version__


def export_csv_columns(source, destination, settings):
    """Project a CSV record at a time, preserving cell text and record order."""
    with (source.open(encoding="utf-8-sig", newline="") as stream,
          destination.open("x", encoding="utf-8", newline="") as output):
        reader = csv.reader(stream, strict=True)
        header, _, error = read_header(reader)
        if error or not header:
            raise ValueError("Cannot read the CSV header")
        counts = Counter(header)
        selected = settings["columns"]
        invalid = [name for name in selected if counts[name] != 1]
        if invalid:
            raise ValueError(f"CSV columns must be present and unambiguous: {', '.join(invalid)}")
        positions = [header.index(name) for name in selected]
        writer = csv.writer(output)
        writer.writerow(selected)
        rows, limit = 0, settings["limit"]
        for row in reader:
            if not row:
                continue
            if len(row) != len(header):
                raise ValueError(f"CSV record width differs near line {reader.line_num}")
            if limit is None or rows < limit:
                writer.writerow([row[position] for position in positions])
                rows += 1
    return rows, [{"name": name, "type": "VARCHAR"} for name in header]


def extract_data(source, output, columns, where=(), limit=None, *, expected_input_sha256=None):
    """Select columns and apply AND filters in source order, writing a reproducible CSV."""
    source = Path(source).resolve()
    settings = extraction_settings(source, columns, where, limit)
    if not source.is_file():
        raise ValueError("Choose a single existing source file")
    digest = file_sha256(source)
    if expected_input_sha256 is not None and digest != expected_input_sha256:
        raise ValueError("The source no longer matches the audited file")
    kind = source.suffix.lower()[1:]
    lineage = None if kind == "csv" else audit_lineage(source, digest, kind)
    output, manifest = prepare(output, SCHEMA, settings)
    manifest.update(input_sha256=digest, source_audit=lineage)
    try:
        with TemporaryDirectory(prefix=".extract-", dir=output) as temporary:
            scratch = Path(temporary)
            csv_path = scratch / CSV_ARTIFACT
            if kind == "csv":
                rows, columns = export_csv_columns(source, csv_path, settings)
                duckdb_version = None
            else:
                rows, columns, duckdb_version = export_csv(source, csv_path, scratch, settings, lineage)
            if file_sha256(source) != digest:
                raise ValueError("The source file changed during extraction")
            csv_hash = file_sha256(csv_path)
            csv_path.rename(output / CSV_ARTIFACT)
        manifest.update(status="completed", rows=rows, source_columns=columns,
                        artifacts={CSV_ARTIFACT: csv_hash}, duckdb_version=duckdb_version)
        write_json(output / "manifest.json", manifest)
        return output
    except Exception as error:
        fail(output, manifest, error)
        raise
