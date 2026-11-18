"""Write an audited CSV to portable, bounded-batch storage."""

from contextlib import ExitStack, closing
import csv
from decimal import Decimal
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from .reporting import file_sha256


CONVERSION_SCHEMA = 16
FORMATS = ("parquet", "duckdb")
TYPE_MODES = ("preserve", "numeric")
BATCH_ROWS = 32_768
BATCH_BYTES = 8 * 1024 ** 2
DUCKDB_CONFIG = {"threads": "1", "memory_limit": "512MB"}


def conversion_settings(formats, types):
    requested = (formats,) if isinstance(formats, str) else formats
    requested = list(dict.fromkeys(requested))
    if any(name not in FORMATS for name in requested):
        raise ValueError("Conversion formats must be parquet or duckdb")
    if types not in TYPE_MODES or (not requested and types != "preserve"):
        raise ValueError("Select a conversion format before requesting numeric conversion types")
    return {"convert": requested, "conversion_types": types} if requested else {}


def conversion_blocker(profile, requested):
    scan = profile["scan"]
    if scan["status"] != "complete" or scan["wrong_width_rows"]:
        return "Repair the CSV structure before converting. The report records the affected input."
    if profile["duplicate_header_names"] or profile["empty_header_columns"]:
        return "Give every column a non-empty and distinct name before converting."
    if "duckdb" in requested:
        ascii_fold = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")
        names = [column["name"] for column in profile["columns"]]
        if any("\0" in name for name in names) or len({name.translate(ascii_fold) for name in names}) != len(names):
            return "DuckDB requires column names that are distinct ignoring ASCII case and contain no NUL."
    return None


def storage_columns(profile):
    columns = []
    for column in profile["columns"]:
        sql = "VARCHAR"
        shape = column.get("decimal_shape")
        if shape and column["observed_type"] == "numeric" and not (column["missing"] or column["nonfinite"]):
            integer_digits, scale = shape
            precision = max(1, integer_digits + scale)
            if scale == 0 and precision <= 18:
                sql = "BIGINT"
            elif precision <= 38:
                sql = f"DECIMAL({precision},{scale})"
        columns.append({"name": column["name"], "type": sql})
    return columns


def arrow_schema(pa, columns, profile):
    def arrow_type(sql):
        if sql == "VARCHAR":
            return pa.string()
        if sql == "BIGINT":
            return pa.int64()
        precision, scale = map(int, re.findall(r"[0-9]+", sql))
        return pa.decimal128(precision, scale)

    return pa.schema(
        [(column["name"], arrow_type(column["type"])) for column in columns],
        metadata={
            "diy_automl_input_sha256": profile["input_sha256"],
            "diy_automl_conversion_types": profile["settings"]["conversion_types"],
        },
    )


def csv_batches(path, names):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        records = (row for row in csv.reader(stream, strict=True) if row)
        if next(records, None) != names:
            raise ValueError("CSV header changed before conversion")
        batch, estimated_bytes = [], 0
        for row in records:
            if len(row) != len(names):
                raise ValueError("CSV record width changed before conversion")
            row_bytes = sum(64 + 4 * len(value) for value in row)
            if batch and (len(batch) >= BATCH_ROWS or estimated_bytes + row_bytes > BATCH_BYTES):
                yield batch
                batch, estimated_bytes = [], 0
            batch.append(row)
            estimated_bytes += row_bytes
        if batch:
            yield batch


def to_arrow(pa, schema, columns, batch):
    def values(raw, sql):
        if sql == "VARCHAR":
            return raw
        if sql == "BIGINT":
            return [int(Decimal(value.strip())) for value in raw]
        return [Decimal(value.strip()) for value in raw]

    return pa.RecordBatch.from_arrays(
        [pa.array(values(raw, column["type"]), type=field.type)
         for raw, column, field in zip(zip(*batch), columns, schema)],
        schema=schema,
    )


def sql_identifier(value):
    return '"' + value.replace('"', '""') + '"'


def engines(requested):
    try:
        import pyarrow as pa
        import pyarrow.parquet as parquet
        if "duckdb" in requested:
            import duckdb
        else:
            duckdb = None
    except ImportError as error:
        raise RuntimeError("Install the optional storage dependencies with pip install -e '.[storage]'") from error
    return pa, parquet, duckdb


def write_duckdb_header(db, columns, profile, rows):
    fields = ", ".join(f"{sql_identifier(column['name'])} {column['type']}" for column in columns)
    db.execute(f"CREATE TABLE data ({fields})")
    db.execute("CREATE TABLE audit_source (input_sha256 VARCHAR, conversion_types VARCHAR, row_count BIGINT)")
    db.execute("INSERT INTO audit_source VALUES (?, ?, ?)",
               (profile["input_sha256"], profile["settings"]["conversion_types"], rows))


def write_formats(csv_path, stage, files, columns, profile):
    pa, parquet, duckdb = engines(files)
    schema = arrow_schema(pa, columns, profile)
    expected_rows = profile["scan"]["valid_rows"]
    written = 0
    with ExitStack() as stack:
        writer = stack.enter_context(parquet.ParquetWriter(files["parquet"], schema, compression="zstd")) if "parquet" in files else None
        db = None
        if "duckdb" in files:
            config = {**DUCKDB_CONFIG, "temp_directory": str(stage / "spill")}
            db = stack.enter_context(closing(duckdb.connect(str(files["duckdb"]), config=config)))
            db.execute("BEGIN TRANSACTION")
            write_duckdb_header(db, columns, profile, expected_rows)
        for batch in csv_batches(csv_path, schema.names):
            arrow = to_arrow(pa, schema, columns, batch)
            if writer is not None:
                writer.write_batch(arrow)
            if db is not None:
                db.register("_audit_batch", pa.Table.from_batches([arrow]))
                try:
                    db.execute("INSERT INTO data SELECT * FROM _audit_batch")
                finally:
                    db.unregister("_audit_batch")
            written += arrow.num_rows
        if written != expected_rows or file_sha256(csv_path) != profile["input_sha256"]:
            raise ValueError("CSV content or record count changed during conversion")
        if db is not None:
            db.execute("COMMIT")


def verify_formats(files, columns, profile):
    pa, parquet, duckdb = engines(files)
    rows = profile["scan"]["valid_rows"]
    if "parquet" in files:
        with closing(parquet.ParquetFile(files["parquet"])) as stored:
            expected = arrow_schema(pa, columns, profile)
            if stored.metadata.num_rows != rows or not stored.schema_arrow.equals(expected, check_metadata=True):
                raise ValueError("Parquet row count, columns, types or source metadata differ")
    if "duckdb" in files:
        config = {**DUCKDB_CONFIG, "enable_external_access": "false"}
        with closing(duckdb.connect(str(files["duckdb"]), read_only=True, config=config)) as db:
            expected = [(column["name"], column["type"]) for column in columns]
            actual = [row[:2] for row in db.execute("DESCRIBE SELECT * FROM data").fetchall()]
            provenance = db.execute("SELECT * FROM audit_source").fetchall()
            if (actual != expected or db.execute("SELECT COUNT(*) FROM data").fetchone()[0] != rows
                    or provenance != [(profile["input_sha256"], profile["settings"]["conversion_types"], rows)]):
                raise ValueError("DuckDB row count, columns, types or source metadata differ")


def converted_path(format):
    return f"converted/data.{format}"


def convert_csv(csv_path, output, profile):
    requested = profile["settings"]["convert"]
    result = {"status": "blocked", "reason": conversion_blocker(profile, requested), "files": {}}
    if result["reason"]:
        return result
    columns = storage_columns(profile)
    try:
        with TemporaryDirectory(prefix=".conversion-", dir=output) as scratch:
            stage = Path(scratch) / "converted"
            stage.mkdir()
            files = {name: stage / f"data.{name}" for name in requested}
            write_formats(csv_path, Path(scratch), files, columns, profile)
            verify_formats(files, columns, profile)
            artifacts = {
                name: {"path": converted_path(name), "bytes": path.stat().st_size, "sha256": file_sha256(path)}
                for name, path in files.items()
            }
            stage.rename(output / "converted")
    except Exception as error:
        return {**result, "status": "failed", "reason": f"{type(error).__name__}: {error}"[:1000]}
    return {"status": "completed", "reason": None, "files": artifacts, "rows": profile["scan"]["valid_rows"],
            "columns": columns}
