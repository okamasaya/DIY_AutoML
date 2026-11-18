"""Measure the normal DIY AutoML CLI on fixed synthetic Windows workloads."""

import argparse
import csv
import ctypes as ct
from ctypes import wintypes as win
from datetime import datetime, timezone
from hashlib import file_digest
from importlib.metadata import version
from itertools import islice
import json
import os
from pathlib import Path
import platform
import runpy
import subprocess
import sys
from time import perf_counter


DATA_SHAPES = ((10_000, 7), (100_000, 7), (100_000, 28), (1_000_000, 7))
TRAIN_SHAPES = ((1_000, 7), (5_000, 7), (10_000, 7), (5_000, 28))
GENERATOR_SEED = 1842
CONDITIONS = ("--where", "x00", "ge", "7500", "--where", "x01", "ge", "5000")
EXTRACT_LIMIT = 10_000
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS")


class ProcessMemoryCounters(ct.Structure):
    _fields_ = [
        ("size", win.DWORD), ("page_fault_count", win.DWORD),
        ("peak_working_set", ct.c_size_t), ("working_set", ct.c_size_t),
        ("peak_paged_pool", ct.c_size_t), ("paged_pool", ct.c_size_t),
        ("peak_nonpaged_pool", ct.c_size_t), ("nonpaged_pool", ct.c_size_t),
        ("pagefile", ct.c_size_t), ("peak_pagefile", ct.c_size_t),
    ]


class MemoryStatus(ct.Structure):
    _fields_ = [
        ("size", win.DWORD), ("load", win.DWORD),
        ("total_physical", ct.c_uint64), ("available_physical", ct.c_uint64),
        ("total_pagefile", ct.c_uint64), ("available_pagefile", ct.c_uint64),
        ("total_virtual", ct.c_uint64), ("available_virtual", ct.c_uint64),
        ("extended_virtual", ct.c_uint64),
    ]


def peak_working_set():
    kernel = ct.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = win.HANDLE
    query = kernel.K32GetProcessMemoryInfo
    query.argtypes = [win.HANDLE, ct.POINTER(ProcessMemoryCounters), win.DWORD]
    query.restype = win.BOOL
    counters = ProcessMemoryCounters()
    counters.size = ct.sizeof(counters)
    if not query(kernel.GetCurrentProcess(), ct.byref(counters), counters.size):
        raise ct.WinError(ct.get_last_error())
    return counters.peak_working_set


def host():
    import winreg

    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
        processor, _ = winreg.QueryValueEx(key, "ProcessorNameString")
    status = MemoryStatus()
    status.size = ct.sizeof(status)
    kernel = ct.WinDLL("kernel32", use_last_error=True)
    kernel.GlobalMemoryStatusEx.argtypes = [ct.POINTER(MemoryStatus)]
    kernel.GlobalMemoryStatusEx.restype = win.BOOL
    if not kernel.GlobalMemoryStatusEx(ct.byref(status)):
        raise ct.WinError(ct.get_last_error())
    return {
        "platform": platform.platform(),
        "processor": processor.strip(),
        "logical_cpus": os.cpu_count(),
        "physical_memory_bytes": status.total_physical,
        "available_physical_at_start_bytes": status.available_physical,
        "python": platform.python_version(),
        "dependencies": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "pyarrow", "duckdb")},
    }


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def fingerprint(path):
    with path.open("rb") as file:
        return {"bytes": path.stat().st_size, "sha256": file_digest(file, "sha256").hexdigest()}


def columns(features):
    return [f"x{index:02d}" for index in range(features)] + ["target"]


def make_input(path, rows, features):
    import numpy as np

    rng = np.random.default_rng(GENERATOR_SEED)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(columns(features))
        for start in range(0, rows, 10_000):
            values = rng.integers(0, 10_000, (min(10_000, rows - start), 28))
            score = values[:, 2:7] @ np.array([4, 3, 2, 1, 1])
            labels = (score >= 55_000).astype(np.int64)
            writer.writerows(np.column_stack((values[:, :features], labels)).tolist())


def csv_prefix(source, destination, rows):
    with source.open("r", newline="", encoding="utf-8") as src, destination.open("w", newline="", encoding="utf-8") as dst:
        row_count = 0
        for line in islice(src, rows + 1):
            dst.write(line)
            row_count += 1
    if row_count != rows + 1:
        raise ValueError(f"Requested {rows} rows but {source} provided {row_count - 1}")


def measure_cli(measurement, cli_args):
    sys.argv = ["diy_automl", *cli_args]
    try:
        runpy.run_module("diy_automl", run_name="__main__")
    finally:
        write_json(Path(measurement), {"peak_working_set_bytes": peak_working_set()})


def measure(folder, operation, command, repeat, env):
    job = folder / f"{operation}-{repeat}"
    job.mkdir()
    output = job / "output"
    stats = job / "memory.json"
    log = job / "console.log"
    args = [*command, "--output", str(output)]
    invocation = [sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "--measure", str(stats), *args]
    with log.open("wb") as log_file:
        start = perf_counter()
        result = subprocess.run(invocation, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        elapsed = perf_counter() - start
    if result.returncode:
        raise RuntimeError(f"{command[0]} exited with {result.returncode}. See {log}")
    memory = read_json(stats)["peak_working_set_bytes"]
    print(f"{folder.relative_to(folder.parent.parent)!s:>20}  {job.name:<16} {elapsed:6.2f} s {memory / 2**20:6.0f} MiB", flush=True)
    return output, {"repeat": repeat, "wall_seconds": elapsed, "peak_working_set_bytes": memory}


def storage_case(root, rows, features, repeats, env):
    folder = root / f"{rows}x{features}"
    folder.mkdir()
    source = folder / "input.csv"
    make_input(source, rows, features)
    evidence = {
        "input_rows": rows, "features": features, "csv": fingerprint(source),
        "audit": [], "audit_parquet": [], "extract": [],
    }
    source_args = ["audit-data", "--csv", str(source), "--target", "target"]
    for repeat in range(1, repeats + 1):
        repeated = folder / f"repeat-{repeat}"
        repeated.mkdir()
        audit, timing = measure(repeated, "audit", source_args, repeat, env)
        evidence["audit"].append(timing)
        converted, timing = measure(repeated, "audit-parquet", [*source_args, "--convert", "parquet",
                                                               "--conversion-types", "numeric"], repeat, env)
        evidence["audit_parquet"].append(timing)
        parquet = converted / "converted" / "data.parquet"
        extracted, timing = measure(
            repeated, "extract", ["extract-data", "--source", str(parquet), "--columns", *columns(features),
                       *CONDITIONS, "--limit", str(EXTRACT_LIMIT)], repeat, env,
        )
        evidence["extract"].append(timing)
        if read_json(audit / "manifest.json")["input_sha256"] != evidence["csv"]["sha256"]:
            raise ValueError(f"Unexpected audit input in {audit}")
    evidence["parquet"] = fingerprint(parquet)
    evidence["extracted_csv"] = fingerprint(extracted / "data.csv")
    evidence["extracted_rows"] = read_json(extracted / "manifest.json")["rows"]
    evidence["paths"] = {label: str(path.relative_to(root)) for label, path in
                         [("audit", audit), ("audit_parquet", converted), ("extract", extracted)]}
    return evidence, extracted / "data.csv"


def training_case(root, rows, features, extracted_csv, repeats, env):
    folder = root / f"select-{rows}x{features}"
    folder.mkdir()
    source = folder / "input.csv"
    csv_prefix(extracted_csv, source, rows)
    evidence = {"input_rows": rows, "features": features, "csv": fingerprint(source), "select": []}
    for repeat in range(1, repeats + 1):
        output, timing = measure(folder, "select", ["select", "--csv", str(source), "--target", "target",
                                                    "--seed", "42", "--bootstrap-repeats", "1000"], repeat, env)
        evidence["select"].append(timing)
    evidence["last_output"] = str(output.relative_to(root))
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if sys.platform != "win32":
        parser.error("This benchmark uses Windows process memory counters")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    root = args.output.resolve()
    root.mkdir(parents=True)
    threads = dict.fromkeys(THREAD_VARS, "1")
    env = dict(os.environ, **threads)
    summary = {
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": host(), "thread_environment": threads, "repeats": args.repeats,
        "generator_seed": GENERATOR_SEED, "extract_conditions": list(CONDITIONS),
        "extract_limit": EXTRACT_LIMIT, "storage": [], "training": [],
    }
    measurement_path = root / "results.json"
    sources = {}
    for rows, features in DATA_SHAPES:
        evidence, sources[features] = storage_case(root, rows, features, args.repeats, env)
        summary["storage"].append(evidence)
        write_json(measurement_path, summary)
    for rows, features in TRAIN_SHAPES:
        evidence = training_case(root, rows, features, sources[features], args.repeats, env)
        summary["training"].append(evidence)
        write_json(measurement_path, summary)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--measure"]:
        measure_cli(sys.argv[2], sys.argv[3:])
    else:
        main()
