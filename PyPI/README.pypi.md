# UC_AutoMaiL

UC_AutoMaiL is a Python package for comparing machine learning models and evaluating the uncertainty of their scores. It started as DIY AutoML. It saves data checks, evaluation splits, model selection, predictions, held-out scores, and their uncertainty in files you can inspect after a run.

## Try the alpha

Use Python 3.12 or later and activate a fresh virtual environment. If you have the `0.4.1a1` wheel, place it in your working directory and install it with dependencies from PyPI. For index-based installation, see the release guide. [Release and TestPyPI instructions](https://github.com/okamasaya/DIY_AutoML/blob/main/PyPI/README.md)

```sh
python -m pip --isolated install --index-url https://pypi.org/simple/ ./uc_automail-0.4.1a1-py3-none-any.whl
python -m pip check
uc-automail --version
uc-automail --help
```

Import the public Python API as `uc_automail`.

```python
import uc_automail as uc_al

print(uc_al.__version__)
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

For streaming observations, construct an EKF once and pass one observation vector at a time. This example uses a one-dimensional state and a known increment at each step.

```python
import uc_automail as uc_al

settings = dict(
    initial_mean=[0.0], initial_covariance=[[1.0]],
    transition=lambda state, step, control: state + control,
    transition_jacobian=lambda state, step, control: [[1.0]],
    measurement=lambda state, step: state,
    measurement_jacobian=lambda state, step: [[1.0]],
    process_covariance=[[0.01]], observation_covariance=[[0.1]],
)
observations = [[1.1], [2.0], [float("nan")], [3.9]]
tracker = uc_al.ExtendedKalmanFilter(**settings)
for observation in observations:
    estimate = tracker.step(observation, control=1.0)
    print(estimate["filtered"])

batch = uc_al.extended_kalman_filter(observations, controls=[1.0] * len(observations), **settings)
diagnostics = uc_al.filter_diagnostics(batch, observations, max_lag=2)
print(diagnostics["autocorrelation_pairs"], diagnostics["ljung_box_status"])
```

`filter_diagnostics` accepts either EKF or UKF batch output. It reports the normalized innovation, NIS, autocorrelations and a per-channel Ljung–Box status. The white-noise reference band and the lags outside it are given for complete channels with varying innovations. When the true state is available, `uc_al.evaluate_filter_consistency(truth, result)` returns pull summaries and NEES. Reported covariance reflects the transition and measurement models and the supplied noise covariances. [State-estimation guide](https://github.com/okamasaya/DIY_AutoML/blob/main/docs/UNCERTAINTY_DESIGN.md)

Existing DIY AutoML examples can still use the `diy_automl` import and the `diy-automl` command. The UC_AutoMaiL Python package is available under the MIT License. [License and scope](https://github.com/okamasaya/DIY_AutoML/blob/main/LICENSE)

[English documentation](https://github.com/okamasaya/DIY_AutoML/blob/main/README.en.md) · [Iris walkthrough](https://dev.to/masaya_oka/audit-an-iris-csv-before-ranking-five-classifiers-457) · [Forecasting examples](https://dev.to/masaya_oka/when-should-you-keep-a-kalman-filter-three-forecasting-checks-a64) · [Report an issue](https://github.com/okamasaya/DIY_AutoML/issues)
