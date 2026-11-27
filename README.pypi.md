# UC_AutoMaiL

UC_AutoMaiL is a Python package for comparing machine learning models and evaluating the uncertainty of their scores. It started as DIY AutoML. It saves data checks, evaluation splits, model selection, predictions, held-out scores, and their uncertainty in files you can inspect after a run.

## Try the alpha

Use Python 3.12 or later. In an active virtual environment, install the alpha version explicitly:

```sh
python -m pip install uc-automail==0.4.0a8
uc-automail --version
uc-automail --help
```

To try classification, download the [Iris sample CSV](https://raw.githubusercontent.com/okamasaya/DIY_AutoML/main/data/reference/iris.csv) and save it as `iris.csv`.

```sh
uc-automail audit-data --csv iris.csv --target target --output runs/iris-audit
uc-automail select --csv iris.csv --target target --progress both --output runs/iris-select
uc-automail report --run runs/iris-select --charts detail --output runs/iris-report
```

Open `runs/iris-report/report_with_figures.xlsx` for the comparison, held-out scores, and charts of cross-validation scores and progress. The `Progress` sheet and `progress.jsonl` retain events from model selection. `arrays.npz` stores the splits and predictions, and `manifest.json` records the settings and file hashes. Choose new output paths when repeating commands.

For supervised classification, `select` accepts per-model tree depth limits such as `--decision-tree-max-depth 3` and CPU controls such as `--cpu-threads 2 --forest-jobs 2`. To prepare a smaller dataset, `extract-data --source iris.csv --recipe columns.json --output runs/iris-extract` reads an ordered `columns` array and an optional `format` from JSON. CSV is the default. Install `uc-automail[storage]` for Parquet and DuckDB input or output.

The base install includes tabular evaluation, Excel reports, state estimation, and graph node classification. The `storage` extra adds Parquet and DuckDB support. The `dl` extra adds TensorFlow for the neural-network workflows. The Japanese Windows preview uses Tcl/Tk and a desktop display.

The Python import is `diy_automl`. Examples written for DIY AutoML can also use the `diy-automl` command. The UC_AutoMaiL Python package is available under the MIT License. [License and scope](https://github.com/okamasaya/DIY_AutoML/blob/main/LICENSE)

[English documentation](https://github.com/okamasaya/DIY_AutoML/blob/main/README.en.md) · [Iris walkthrough](https://dev.to/masaya_oka/audit-an-iris-csv-before-ranking-five-classifiers-457) · [Forecasting examples](https://dev.to/masaya_oka/when-should-you-keep-a-kalman-filter-three-forecasting-checks-a64) · [Report an issue](https://github.com/okamasaya/DIY_AutoML/issues)
