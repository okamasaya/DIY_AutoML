# DIY AutoML

DIY AutoML is a Python package for comparing machine learning models and examining the evidence behind their scores. It records data checks, evaluation splits, model selection, predictions, and held-out performance in files that can be inspected after a run.

## Try the alpha

Use Python 3.12 or later. In an active virtual environment, install the alpha version explicitly:

```sh
python -m pip install diy-automl==0.4.0a8
diy-automl --version
diy-automl --help
```

To try classification, download the [Iris sample CSV](https://raw.githubusercontent.com/okamasaya/DIY_AutoML/main/data/reference/iris.csv) and save it as `iris.csv`.

```sh
diy-automl audit-data --csv iris.csv --target target --output runs/iris-audit
diy-automl select --csv iris.csv --target target --output runs/iris-select
```

Open `runs/iris-select/report.xlsx` to see the comparison and held-out scores. `arrays.npz` stores the splits and predictions, and `manifest.json` records the settings and file hashes. Choose new output paths when repeating commands.

For supervised classification, `select` accepts per-model tree depth limits such as `--decision-tree-max-depth 3` and CPU controls such as `--cpu-threads 2 --forest-jobs 2`. To prepare a smaller dataset, `extract-data --source iris.csv --recipe columns.json --output runs/iris-extract` reads an ordered `columns` array and an optional `format` from JSON. CSV is the default. Install `diy-automl[storage]` for Parquet and DuckDB input or output.

The base install includes tabular evaluation, Excel reports, state estimation, and graph node classification. The `storage` extra adds Parquet and DuckDB support. The `dl` extra adds TensorFlow for the neural-network workflows. The Japanese Windows preview uses Tcl/Tk and a desktop display.

[English documentation](https://github.com/okamasaya/DIY_AutoML/blob/main/README.en.md) · [Iris walkthrough](https://dev.to/masaya_oka/audit-an-iris-csv-before-ranking-five-classifiers-457) · [Forecasting examples](https://dev.to/masaya_oka/when-should-you-keep-a-kalman-filter-three-forecasting-checks-a64) · [Report an issue](https://github.com/okamasaya/DIY_AutoML/issues)
