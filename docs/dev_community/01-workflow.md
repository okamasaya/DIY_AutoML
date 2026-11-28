---
title: "Audit an Iris CSV before ranking five classifiers"
published: false
description: "Install UC_AutoMaiL from TestPyPI, run an Iris classification workflow in Python, and inspect the results in Excel."
tags: python, machinelearning, datascience, scikitlearn
series: "DIY AutoML Development Notes"
ai_disclosure_level: some_ai
---

I ran five classifiers on Iris and selected an SVM. The held-out macro F1 was 0.967, with a conditional 95% interval of 0.898 to 1.000. To decide what that result is worth, I want to know which data went in, how the candidates were compared, and which rows produced the final score. DIY AutoML saves those details from the initial audit through the evaluation.

The Python package from this project is available as **UC_AutoMaiL**. This walkthrough uses the `0.4.1a0` alpha and was checked on Windows with Python 3.12.16 and 3.13.17. It generates Iris from scikit-learn, compares five classifiers on a CPU, and saves an Excel workbook with the holdout results. [TestPyPI release](https://test.pypi.org/project/uc-automail/0.4.1a0/)

## Trace the result back to the input

![Keep and audit the source CSV. Prepare the CSV directly or use a Parquet or DuckDB working copy. Set aside the holdout, compare models on the training split, then evaluate on the holdout and save the results for reanalysis and Excel reporting.](https://raw.githubusercontent.com/okamasaya/DIY_AutoML/8df22d59dbd5378d52edf87f3aeb3ca55938e86e/docs/dev_community/images/audit-to-evaluation.png)

The audit records issues such as missing values and duplicate rows. If a dataset will be read and filtered several times, a Parquet or DuckDB working copy can make that preparation easier to repeat. After extracting the training data, check the row count, missing values, and label counts again. For the 150-row Iris example, the original CSV is small enough to use directly. [Workflow in Figure 1 of the full paper (Japanese)](https://github.com/okamasaya/DIY_AutoML/blob/64b17f120dad1ead6256d963b779c13630457898/docs/papers/DIY_AutoML_Full_Paper_20261119/DIY_AutoML_Full_Paper_20261119.pdf)

## Install UC_AutoMaiL from TestPyPI

Use Python 3.12 or newer. In a new working directory, create and activate a virtual environment. On Windows PowerShell, run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux with bash or zsh, run:

```sh
python3 -m venv .venv
source .venv/bin/activate
```

The first command downloads the pinned wheel from TestPyPI. The second installs that local file and resolves its dependencies from the regular PyPI index. [Python Packaging User Guide](https://packaging.python.org/en/latest/guides/using-testpypi/#using-testpypi-with-pip)

```sh
python -m pip --isolated download --index-url https://test.pypi.org/simple/ --no-deps --only-binary=:all: uc-automail==0.4.1a0
python -m pip --isolated install --index-url https://pypi.org/simple/ ./uc_automail-0.4.1a0-py3-none-any.whl
python -m pip check
python -m uc_automail --version
python -c "import uc_automail as uc_al; print(uc_al.__version__)"
```

Expect `UC_AutoMaiL 0.4.1a0` from the CLI and `0.4.1a0` from the Python import. If PowerShell blocks the activation script, use `.\.venv\Scripts\python.exe` in place of `python` for the remaining commands.

## Inspect Iris and compare five classifiers

Save this as `iris_demo.py`. It writes Iris to CSV, audits the input, compares the candidates, and renders the selected model's evaluation as a workbook. Each run gets its own timestamped directory. [Iris dataset in scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)

```python
from datetime import datetime
from pathlib import Path

import uc_automail as uc_al
from sklearn.datasets import load_iris

work = Path("runs") / datetime.now().strftime("iris-%Y%m%d-%H%M%S-%f")
work.mkdir(parents=True)
csv_path = work / "iris.csv"
load_iris(as_frame=True).frame.to_csv(csv_path, index=False)

uc_al.audit_data_csv(csv_path, output=work / "audit", target="target", row_unit="One iris plant")
selection = uc_al.select_csv(
    csv_path, target="target", output=work / "selection", seed=42, progress="both"
)
report = uc_al.render_excel_report(selection, output=work / "report", charts="detail")
print(report / "report_with_figures.xlsx")
```

```sh
python iris_demo.py
```

The audit found 50 rows for each of the three classes and one exact duplicate across all columns. An identical row prompts a check of the original records to see whether two rows describe the same observation. I kept all 150 rows for this walkthrough.

The split put 120 rows on the training side and reserved 30 for the final test. A decision tree, random forest, logistic regression, support vector machine (SVM), and k-nearest neighbors all used the same five cross-validation splits on the 120 rows. Standardization was fitted on the training portion of each fold. Candidate scores use macro F1, which averages the F1 scores of the classes with equal weight.

| What I checked | Result |
| --- | --- |
| Selected model and mean cross-validation macro F1 | SVM, 0.967 |
| Performance on the 30 held-out rows | 29 correct, macro F1 of 0.967 |
| Conditional 95% interval for held-out macro F1 | 0.898 to 1.000 |

The two F1 values round to the same number and come from different rows. The cross-validation average selects the candidate. The held-out score evaluates that candidate after refitting it on all 120 training rows. The interval shows how the score varies when the evaluation rows are resampled within their recorded classes with the fitted model held fixed. [Evaluation guide](https://github.com/okamasaya/DIY_AutoML/blob/64b17f120dad1ead6256d963b779c13630457898/docs/DATA_SCIENCE_GUIDE.en.md#interpret-an-uncertainty-estimate)

## Hand off the same result in two ways

Open `audit/report.xlsx` to check the input. Open `report/report_with_figures.xlsx` to see the evaluation and its four figures. Start with `SL_CV_Mean` for candidate selection, `SL_Test` for the holdout scores and confusion matrix, and `SL_Test_Intervals` for the score intervals. `progress="both"` puts fold-by-fold updates in the terminal, `selection/progress.jsonl`, and the workbook's `Progress` sheet.

`selection/arrays.npz` holds the splits and predictions, and `selection/manifest.json` records the settings and input hash. A team can use the scores from the same folds to quantify the trade-off when it chooses a model lower in the ranking for operational reasons. [Data science guide](https://github.com/okamasaya/DIY_AutoML/blob/64b17f120dad1ead6256d963b779c13630457898/docs/DATA_SCIENCE_GUIDE.en.md) · [Business user guide](https://github.com/okamasaya/DIY_AutoML/blob/64b17f120dad1ead6256d963b779c13630457898/docs/BUSINESS_USER_GUIDE.en.md)

To try your own CSV, point `csv_path` at a file with numeric feature columns and set `target` to its label column in both calls. Choose a holdout that reflects the records you expect the model to handle. [DIY AutoML source and issues](https://github.com/okamasaya/DIY_AutoML)

## What I want to measure next

The next experiments cover library cross-checks, reusable data preparation, feature selection, reading errors in Excel, and time-series forecasting.

For date-ordered classification, I want to estimate how uncertain the gap between a model and a simple rule is when nearby days are dependent. Then I want to see how results change with the amount of training data and when the entire selection process is repeated. [Project roadmap (Japanese)](https://github.com/okamasaya/DIY_AutoML/blob/64b17f120dad1ead6256d963b779c13630457898/docs/ROADMAP.md)

*AI assistance was used to adapt the project's development notes and experiment records into this English article.*
