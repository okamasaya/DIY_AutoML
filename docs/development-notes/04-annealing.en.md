# Testing Feature Subsets with Simulated Annealing

After preparing a dataset, I sometimes want to know which measurements I can leave out. If a column is expensive to collect, a model that uses fewer columns could make the next round of data collection easier. [Series index](README.md) · [Japanese original](../../articles/diy-automl-development-note-04-annealing.md)

There are 4,095 nonempty ways to choose up to six of Wine's 13 features. Evaluating each combination takes a series of model fits. DIY AutoML's optional CPU command, `annealing-check`, sets a budget for how many subsets to score. [Command and report design (Japanese)](../ANNEALING_CHECK.md)

## Give a lower-scoring candidate a chance

Simulated annealing starts from a subset. A nearby candidate adds a column, removes a column, or swaps a selected column with an unselected one. The search accepts a candidate whose score is at least as high as the current one. It sometimes accepts a lower-scoring candidate, which lets it move through a worse subset to look for a better one. [Original paper on simulated annealing](https://doi.org/10.1126/science.220.4598.671)

For a score decrease, the acceptance probability is `exp((candidate score - current score) / temperature)`. The temperature falls from 0.03 to 0.002 by default. For example, a drop of 0.01 has a roughly 72% chance of acceptance at the initial temperature and about 0.67% at the final temperature. The report records the moves and returns the best subset found, preferring fewer columns when scores tie. [Feature subset search implementation](../../src/diy_automl/annealing.py)

## Compare searches with the same number of candidates

I fixed the classifier to an SVM and changed the feature subsets. All candidates used the same two repeats of stratified five-fold cross-validation on the training data, with the scaler fitted on each fold's training portion. The search scored candidates by mean fold macro F1.

Annealing and random search shared an initial subset and evaluated the same number of distinct subsets. Random search drew uniformly from the remaining eligible combinations. Each candidate required ten fits, so a budget of eight meant 80 fits per method, and a budget of 32 meant 320. I also ran a separate model with all features and evaluated all 15 nonempty subsets on Iris.

Each method refitted the classifier on all the training rows using its chosen subset. The final tests used the same held-out rows, 30 for Iris and 36 for Wine. I kept the data splits fixed, varying only the search seed from 44 through 53.

| Dataset | Budget per method | Cross-validation result |
| --- | ---: | --- |
| Iris | 8 subsets | The exhaustive maximum was 0.9707, using all four features. Annealing reached it in 3 of 10 runs and random search in 8 of 10. |
| Wine | 32 subsets | With up to six features per subset, each search method had the higher cross-validation score in 5 of 10 runs. |

On Wine at seed 44, the five-feature annealing model and the full 13-feature model agreed on all 36 predictions. This happened in one of the ten runs. In that run, annealing scored **+0.0558** in held-out macro F1 relative to random search, with a 95% interval of **[0.0000, +0.1419]**.

The interval keeps the fitted models fixed and resamples the same evaluation rows for both methods within classes. The intervals included zero for all 20 Iris and Wine searches, leaving the direction unresolved under these conditions. The runs show how the subset choice can vary even with the same split and evaluation budget. [Experiment settings and results (Japanese)](../ANNEALING_CHECK.md#irisとwineでの検証)

I replayed the searches on the same 15 Iris cross-validation scores. Increasing the budget from 8 to 12 raised the default annealing schedule's optimum hit count from 3 to 6 out of 10 seeds. Random search hit it 8 times at each budget. Ending at a warmer temperature (0.01 instead of 0.002) gave 3 and 7 hits, respectively. Under this schedule, changing the budget changes both the number of proposals and the rate of cooling per step. These results describe one fixed Iris split. If measurement costs differ, the number of columns is a starting point for comparing them. [Follow-up settings and results (Japanese)](../ANNEALING_CHECK.md#irisで探索予算と終点温度を変える)

## Reproduce search seed 44

From the repository root, run these PowerShell commands using the installed `.venv`. Use a new value for `$work` on subsequent runs.

```powershell
$python = ".\.venv\Scripts\python.exe"
$work = "runs/en-note-04-annealing"

& $python -X utf8 -m diy_automl annealing-check --csv data/reference/iris.csv --target target --budget 8 --exhaustive --search-seed 44 --output "$work/iris"
& $python -X utf8 -m diy_automl annealing-check --csv data/reference/wine.csv --target target --budget 32 --max-features 6 --search-seed 44 --output "$work/wine"
```

In `report.xlsx`, `Selected_Features` shows the chosen columns, `Search_History` shows the scores and accepted moves, and `Test_Differences` shows the held-out score differences and intervals. If you are considering dropping a measurement, the saved predictions can show which cases would be classified differently. [Next note on the Windows preview](05-business-preview.en.md)
