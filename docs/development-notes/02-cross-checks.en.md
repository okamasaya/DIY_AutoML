# Cross-Checking with TensorFlow, PyTorch, AutoGluon, and H2O AutoML

I used Iris and Wine to ask two questions. How closely do two neural-network implementations agree under matched conditions? How do separately selected models compare when evaluated on the same held-out rows? [Series index](README.md) · [Japanese original](../../articles/diy-automl-development-note-02-cross-check.md)

## Compare labels and probabilities across two neural-network libraries

I used TensorFlow 2.20.0 and PyTorch 2.14.0 on a Windows CPU with Python 3.13.17. Both networks had hidden layers of 32, 16, and 8 units. They shared the standardization fitted on the training rows, Float32 inputs, initial weights, and minibatch order. I ran three training seeds. When both libraries used the same weights for inference, the largest absolute difference in predicted probabilities on the 0–1 scale was about 6 × 10⁻⁸. [Experiment settings and reproduction steps](../../experiments/dl_backend_crosscheck/README.md)

After training each implementation separately, the predicted labels matched row for row on the holdout for all three seeds. The confusion matrices and macro F1 also matched. The probabilities showed small differences.

| Dataset | Held-out rows | Largest probability difference after training |
| --- | ---: | ---: |
| Iris | 30 | 3.05 percentage points |
| Wine | 36 | 0.21 percentage points |

For Iris at seed 42, both models predicted class 2 for CSV line 58, counting the header as line 1. TensorFlow assigned class 2 a probability of **56.867%** and PyTorch assigned **53.814%**. The correct class was 1, to which they assigned **42.232%** and **45.253%**, respectively. The confusion matrix records the same error either way.

Log loss reflects the probability assigned to the correct class. For Wine, I kept the fitted models and the 36 held-out rows fixed, then used the same resampled rows for both libraries to estimate the difference in mean log loss across the three seeds (PyTorch minus TensorFlow). When the number of draws increased from 1,000 to 20,000, the upper end of the 95% interval moved from −3.0 × 10⁻⁷ to +1.8 × 10⁻⁷. With the longer run, the interval just included zero. I left the direction of the difference unresolved. [Log loss definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html) · [Recomputing the interval from saved predictions (Japanese)](../../articles/backup/diy-automl-ds-backend-probabilities.md)

## Evaluate AutoGluon on the same splits

AutoGluon Tabular 1.6.3 reused the cross-validation and final test row indices saved by DIY AutoML's `rank-sl` command. Each workflow kept its own candidates and preprocessing. AutoGluon's five candidates in this run were random forest, Extra Trees, k-nearest neighbors, logistic regression, and a weighted ensemble. An available-memory check led to skipping LightGBM on this PC.

Each workflow selected its leader using macro F1 on the shared cross-validation splits. DIY AutoML selected an SVM for both datasets. AutoGluon selected k-nearest neighbors for Iris and logistic regression for Wine. I then scored the winners on the same final test rows. [AutoGluon experiment and reproduction steps](../../experiments/autogluon_sl_crosscheck/README.md)

| Dataset | DIY AutoML F1 | AutoGluon F1 | AutoGluon minus DIY and 95% interval |
| --- | ---: | ---: | --- |
| Iris, 30 rows | 0.9666 | 1.0000 | +0.0334 [0.0000, +0.1023] |
| Wine, 36 rows | 0.9710 | 0.9710 | 0.0000 [−0.0911, +0.0911] |

The paired bootstrap held the fitted models fixed and drew the same held-out rows for both models, within classes, 1,000 times. Both intervals included zero. On Wine, the macro F1 and confusion matrices matched, but the error occurred on CSV line 132 for DIY AutoML and line 136 for AutoGluon, counting the header as line 1.

## Repeat the held-out comparison with H2O AutoML

On the same Windows PC, I ran H2O AutoML 3.46.0.12 with Python 3.14.7 and Java 17. The run used up to five candidates from generalized linear models, distributed random forests, and gradient boosting. H2O used the first of DIY AutoML's saved five-fold repeats and selected by mean fold macro F1. The DIY AutoML models in the comparison came from the original five repeats. H2O selected a generalized linear model on both datasets. [H2O configuration and reproduction steps](../../experiments/h2o_sl_crosscheck/README.md)

| Dataset | H2O AutoML F1 | H2O minus DIY and 95% interval |
| --- | ---: | --- |
| Iris, 30 rows | 1.0000 | +0.0334 [0.0000, +0.1023] |
| Wine, 36 rows | 0.9407 | −0.0302 [−0.0960, 0.0000] |

AutoGluon and H2O each got all 30 Iris rows right. On Wine, DIY AutoML got 35 of 36 right and H2O got 34. The H2O differences were evaluated using the same resampled test rows as the AutoGluon comparison. Both intervals included zero under this evaluation.

## Keep the predictions that explain the score

A workbook gives you the score. Saved row indices, labels, and predictions let you see where two runs agree and where their errors differ. With saved probabilities, I could extend the neural-network comparison to log loss without retraining. That is the part I want DIY AutoML to make easy to follow, from checking the original data to handing off results someone else can read and reproduce. [Next note on Parquet and DuckDB](03-parquet-duckdb.en.md)
