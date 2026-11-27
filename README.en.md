# UC_AutoMaiL Overview

[Japanese README](README.md)

UC_AutoMaiL, originally called DIY AutoML, helps you compare machine learning models and evaluate the uncertainty of their scores. It records which data you used, how you split it, how the candidates ranked, and how the selected model performed on held-out data. The saved predictions and evaluation intervals help you examine individual errors and judge how much the data supports a reported score.

- **Data scientists** can use the Python API, CLI, or notebook to run and reproduce evaluations. [Guide for data scientists](docs/DATA_SCIENCE_GUIDE.en.md) · [Notebook](notebooks/Local_Comparison.ipynb)
- **Business users** can use the Windows preview to inspect a CSV, select columns, compare models, and review results. The preview and its Excel summary sheets currently use Japanese. They show counts such as correct classifications, missed cases, and false alarms. [Guide for business users](docs/BUSINESS_USER_GUIDE.en.md)

The five development notes cover the workflow, library cross-checks, storage benchmarks, feature selection, and the Windows preview. [English development notes](docs/development-notes/README.md)

The distribution trial for version `0.4.0a8` includes a wheel and a CPU notebook for trying classification, state estimation, and graph classification. [Trial instructions (Japanese)](PyPI/TRIAL.md) · [Colab and Jupyter notebook](PyPI/Alpha_Trial.ipynb) · [PyPI release files](PyPI/)

## What you can do

| Task | What UC_AutoMaiL provides |
| --- | --- |
| Inspect a dataset | Check a CSV for missing values, duplicate rows, and class counts. Optionally make a reusable Parquet or DuckDB working copy, then extract a training CSV. [Data preparation (Japanese)](docs/DATA_AUDIT.md) |
| Compare classifiers | Use the same training folds for five scikit-learn classifiers, rank them by mean macro F1 by default, and evaluate the selected model on held-out data. The `select` workflow supports stratified row splits, group splits, and date-ordered splits. |
| Examine uncertainty | Inspect bootstrap intervals for a fixed fitted model on row-based or grouped holdouts. The `rank-sl` and `audit-uncertainty` workflows also check floating-point effects and sensitivity to hypothetical feature offsets. [Uncertainty checks (Japanese)](docs/SL_UNCERTAINTY_AUDIT.md) |
| Explore feature subsets | Run an optional CPU check that compares simulated annealing and random search with the same number of evaluated subsets. [Annealing check (Japanese)](docs/ANNEALING_CHECK.md) |
| Review and reproduce results | Open evaluation workbooks or inspect saved arrays and run manifests. The Windows preview also shows a model comparison chart and, for date-ordered evaluations, monthly scores alongside simple baselines. [Output guide (Japanese)](docs/EXCEL_REPORT_GUIDE.md) |

The repository also includes exploratory clustering and a reference neural network trained with TensorFlow. PyTorch, AutoGluon, and H2O AutoML are used in separate cross-check experiments. [TensorFlow and PyTorch cross-check (Japanese)](experiments/dl_backend_crosscheck/README.md) · [AutoGluon cross-check (Japanese)](experiments/autogluon_sl_crosscheck/README.md) · [H2O AutoML cross-check (Japanese)](experiments/h2o_sl_crosscheck/README.md)

## Try it on Iris

The core package requires Python 3.12 or later. The example uses Python 3.13 and runs on a CPU with the core dependencies. Clone the repository and run the commands from its root.

```sh
git clone https://github.com/okamasaya/DIY_AutoML.git
cd DIY_AutoML
```

### Windows PowerShell

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -X utf8 -m diy_automl audit-data --csv data/reference/iris.csv --target target --output runs/iris-audit
.\.venv\Scripts\python.exe -X utf8 -m diy_automl select --csv data/reference/iris.csv --target target --output runs/iris-select
```

### macOS and Linux shell

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m diy_automl audit-data --csv data/reference/iris.csv --target target --output runs/iris-audit
.venv/bin/python -m diy_automl select --csv data/reference/iris.csv --target target --output runs/iris-select
```

To add Parquet and DuckDB support, use `-e '.[storage]'` in place of `-e .` in the install command. The `dl` extra adds TensorFlow for the neural-network workflow.

The audit writes a workbook and `profile.json` to `runs/iris-audit`. In `runs/iris-select/report.xlsx`, open `SL_CV_Mean` for the comparison of candidates, `SL_Test` for the held-out score, and `SL_Test_Intervals` for its interval. The same run directory contains `arrays.npz` with split indices and predictions and `manifest.json` with settings and file hashes. Use fresh output directories when repeating the commands.

In a Windows run, the workflow used 120 Iris rows to compare candidates, selected an SVM, and classified 29 of the 30 held-out rows correctly. The held-out macro F1 was 0.967. [Iris example and evaluation conditions (Japanese)](articles/diy-automl-development-note-01-overview.md)

Use `--tree-max-depth 3` to cap the depth of the Decision Tree and each Random Forest tree in `run` with SL, `select`, `rank-sl`, or `tune`. The Python keyword is `tree_max_depth=3`. With `tune`, the depth is fixed and the other parameters remain in the search. The specified limit is recorded in the Excel `Summary` and the run manifest. [Tuning guide (Japanese)](docs/SL_TUNING_GUIDE.md)

Use `--decision-tree-max-depth` and `--random-forest-max-depth` for separate limits. Each option takes precedence over the shared limit for its model. The Python equivalent is `tree_max_depths={"Decision Tree": 3, "Random Forest": 5}`. If `tune` receives only one model's limit, it keeps the default depth grid for the other model.

Use `--cpu-threads 2 --forest-jobs 2` to set native thread limits and Random Forest parallelism for supervised evaluation. The Python keywords are `cpu_threads` and `forest_jobs`. The preview uses one native thread and one forest worker for the default CPU evaluation. [Preview setup and MacBook Air target (Japanese)](docs/DATA_AUDIT.md#データ点検の画面を使う)

To open the Windows preview with the same sample, run the command below and choose `target` as the label column. [Preview guide (Japanese)](docs/DATA_AUDIT.md#データ点検の画面を使う)

```powershell
.\.venv\Scripts\python.exe -X utf8 -m diy_automl bp-preview --csv data/reference/iris.csv
```

## Scope and limitations

- Training expects a UTF-8 CSV with a class label column and finite numerical features. Handle missing values and encode categorical features before training. Parquet and DuckDB serve as working formats during data preparation. The training commands consume CSV.
- A bootstrap interval describes evaluation sample variability for the selected fitted model. Variation from retraining, model selection, or changes in the population requires additional evaluation. Date-ordered runs currently report point estimates for the held-out period.
- The Windows preview runs from the Python checkout with Tcl/Tk. A packaged installer and interactive chart editing are future work. [Roadmap (Japanese)](docs/ROADMAP.md)

The current checkout is an alpha version. The project-authored Python package and its packaging files are available under the MIT License. The dataset notes document provenance and terms for the bundled examples. [License and scope](LICENSE) · [Dataset notes (Japanese)](data/reference/README.md)

## Technical letter

The English edition reports the experiments and evaluation methods documented for development version `0.4.0a7` at snapshot `80b25cc`. [Technical letter (PDF)](docs/letters/DIY_AutoML_Letter_20261120_en.pdf) · [LaTeX source](docs/letters/DIY_AutoML_Letter_20261120_en.tex)
