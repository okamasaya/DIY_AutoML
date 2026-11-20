# Reusing Working Data with Parquet and DuckDB

As a CSV grows, repeatedly reading it, parsing text, and pulling out columns takes more time. DIY AutoML can save an audited dataset as a reusable Parquet or DuckDB file. The goal is to reduce storage I/O and repeated CSV parsing so that preparing data for the next run takes less work. [Series index](README.md) · [Japanese original](../../articles/diy-automl-development-note-03-parquet-duckdb.md)

## Keep the original CSV and record the working copy

The workflow keeps and audits the original CSV, optionally creates a working file, then extracts and checks the training data. The audit looks at row counts, missing values, duplicate rows, and label counts. A conversion records the output row count, column types, and SHA-256 hash of the source CSV.

- **Parquet** stores data by column. DIY AutoML writes it with Zstandard compression to reduce the bytes stored and read. [Apache Parquet overview](https://parquet.apache.org/docs/overview/)
- **DuckDB** reads Parquet directly and can limit reads to the projected columns. If the row group statistics show that a group cannot match a filter, DuckDB can skip it. A persistent `.duckdb` file can also hold a table for reuse. [DuckDB Parquet documentation](https://duckdb.org/docs/data/parquet#partial-reading)

For a related benchmark, PKSHA's engineering blog compares file size and query time for CSV, Parquet, and ORC using the same DuckDB queries. Its diagrams show how reading selected columns can save I/O. [PKSHA's DuckDB experiment (Japanese)](https://zenn.dev/pksha/articles/038bcf47e627c2)

By default, conversion preserves the text read from each CSV field. That keeps leading zeros in identifiers, for example. The `--conversion-types numeric` option gives numerical types to feature columns whose values can be represented exactly. [Conversion and verification rules (Japanese)](../DATA_AUDIT.md#parquetとduckdbへの変換)

## Select columns from Raisin while keeping every row

The Raisin CSV has 900 records. I extracted area (`Area`), perimeter (`Perimeter`), and the recorded variety (`target`). Run the PowerShell commands from the repository root. If needed, install the optional dependencies with `.\.venv\Scripts\python.exe -m pip install -e ".[storage]"`.

```powershell
$python = ".\.venv\Scripts\python.exe"
$work = "runs/en-note-03-raisin"

& $python -X utf8 -m diy_automl audit-data --csv data/reference/raisin.csv --target target --convert parquet --convert duckdb --output "$work/source-audit"
& $python -X utf8 -m diy_automl extract-data --source "$work/source-audit/converted/data.parquet" --columns Area Perimeter target --output "$work/from-parquet"
& $python -X utf8 -m diy_automl audit-data --csv "$work/from-parquet/data.csv" --target target --output "$work/extracted-audit"
```

The first command creates both working files. You can request a single format by passing one `--convert` option. To extract from DuckDB, change the input to `data.duckdb` and give the extraction a fresh output directory. Set a new `$work` path to repeat the whole example.

For both formats, the three columns and row order matched the original CSV. The counts stayed at **450 Besni and 450 Kecimen**. The extracted `data.csv` is the file that goes to training. The audit and extraction manifests let you trace the selected columns back to the original data. [Raisin source and example (Japanese)](../../data/reference/README.md)

## Measure each stage on a PC with 16 GB of RAM

I used a Windows PC with a Core i5-14400 and 16 GB of RAM. The test dataset was a synthetic one-million-row table with seven numeric features and a binary label. The CSV occupied **36.2 MB** and the compressed Parquet file **18.9 MB**. For each stage, elapsed time is the median of three runs. Memory is the highest peak working set of the Python process across those runs.

| Stage | Input or output size | Time (s) | Memory (MiB) |
| --- | --- | ---: | ---: |
| Audit CSV | 1,000,000 input rows | 63.3 | 169 |
| Audit and convert to Parquet | 1,000,000 input rows | 91.1 | 186 |
| Extract from the existing Parquet file | 10,000 output rows | 2.3 | 186 |
| Compare and evaluate five models | 10,000 extracted rows | 10.4 | 195 |

The first conversion includes the audit and the write to Parquet. The extraction timing starts with the working file already in place. For this benchmark, two filters and a 10,000-row cap defined the extraction, and the models trained on the extracted dataset. At 100,000 rows, increasing the number of features from seven to 28 raised the audit-plus-conversion time from 7.4 to 15.6 seconds. [Benchmark procedure](../../experiments/tabular_scale_benchmark/README.md) · [All timings and input sizes](../../experiments/tabular_scale_benchmark/results-2026-11-19.json)

For an evaluation, I start with the rows and columns needed to answer the question and generally keep every row in that scope. Around 100,000 rows or 20 columns is a starting point for considering a working file, especially if I will read or reshape the data repeatedly. The cost of training still depends on the rows, features, and models passed to that stage. [Next note on feature selection](04-annealing.en.md)
