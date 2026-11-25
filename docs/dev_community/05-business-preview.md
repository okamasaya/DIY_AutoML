---
title: "Read a model's mistakes in Excel and a Windows preview"
published: false
description: "Follow an Iris evaluation and use a bike rental example to see how missed cases and false alarms affect a decision."
tags: machinelearning, python, dataanalysis, excel
series: "DIY AutoML Development Notes"
ai_disclosure_level: some_ai
---

In a bike rental experiment, a random forest missed eight fewer regular-demand cases than a simple calendar rule. It also produced 28 more false alarms for that class. I wanted a Windows preview that makes those trade-offs visible to someone using Excel. [Japanese original](https://github.com/okamasaya/DIY_AutoML/blob/64b17f120dad1ead6256d963b779c13630457898/articles/diy-automl-development-note-05-bp-preview.md)

## Open Iris and compare five models

Use Python with Tcl/Tk and install DIY AutoML in `.venv`. From the repository root, run this PowerShell command to open the preview with Iris selected. The preview currently uses Japanese labels, which are shown alongside their English meanings below. [Setup and complete walkthrough](https://github.com/okamasaya/DIY_AutoML/blob/64b17f120dad1ead6256d963b779c13630457898/docs/BUSINESS_USER_GUIDE.en.md#try-the-windows-preview-with-iris)

```powershell
.\.venv\Scripts\python.exe -X utf8 -m diy_automl bp-preview --csv data/reference/iris.csv
```

1. Select `target` as the outcome column and click **点検する** (*Inspect data*). Check missing values, duplicates, and class counts. To load another UTF-8 CSV, use **CSVを選ぶ…** (*Choose CSV*).
2. Click **使う列を選ぶ…** (*Choose columns*), keep the four Iris measurements, and click **抽出して再点検** (*Extract and inspect again*). The preview saves a separate training CSV with the selected columns and the outcome. With the optional storage dependencies, **作業用Parquetも保存** (*Save a working Parquet copy*) can save a reusable copy for larger inputs.
3. Click **5モデルを比較する** (*Compare five models*). For Iris, the preview used 120 rows to select an SVM and 30 held-out rows to test it.

Model comparison requires finite numeric features, complete values, and recorded class labels. Before inspection, select **IDでまとめる** (*Group by ID*) to keep a customer's repeated rows together, or **日付順で分ける** (*Split by date*) to evaluate later observations. [Input and split options (Japanese)](https://github.com/okamasaya/DIY_AutoML/blob/64b17f120dad1ead6256d963b779c13630457898/docs/DATA_AUDIT.md#画面からのモデル比較)

## Look at the counts behind a score

On Iris, the result showed **30 evaluated, 29 correct, and one error**. **評価Excelを開く** (*Open evaluation workbook*) opens `bp_report.xlsx`. Start with **まず読む** (*Start here*) for the totals and **ラベル別** (*By outcome*) for missed cases and false alarms. The model comparison bar chart under **グラフを見る…** (*View charts*) shows training-side validation scores. Holdout scores appear on the results screen and in the workbook. [How to read the workbook](https://github.com/okamasaya/DIY_AutoML/blob/64b17f120dad1ead6256d963b779c13630457898/docs/BUSINESS_USER_GUIDE.en.md#read-the-evaluation-workbook)

![The Iris SVM classified 29 of 30 held-out rows correctly. One versicolor was classified as virginica. Macro F1 was 0.967, with a conditional 95% interval of 0.898 to 1.000.](https://raw.githubusercontent.com/okamasaya/DIY_AutoML/8df22d59dbd5378d52edf87f3aeb3ca55938e86e/docs/dev_community/images/iris-report.png)

*English summary redrawn from the saved Iris example workbook.* [Original workbook and evaluation conditions (Japanese)](https://github.com/okamasaya/DIY_AutoML/blob/64b17f120dad1ead6256d963b779c13630457898/docs/BP_OUTPUT_EXAMPLE.md)

Date-ordered evaluations add a monthly holdout chart. In Bike Sharing, I defined 200 or more rentals in a recorded hour as busy, and anything below that as regular. I held out the latest fifth of observed dates, which gave 582 rows from August 7 to December 31, 2012. I compared the selected random forest with a simple rule that predicts the most frequent training label for the same day of the week and hour. [Bike Sharing dataset](https://archive.ics.uci.edu/dataset/275/bike%2Bsharing%2Bdataset) · [Data preparation and split (Japanese)](https://github.com/okamasaya/DIY_AutoML/blob/64b17f120dad1ead6256d963b779c13630457898/docs/DATA_AUDIT.md#日付順の分割)

![Monthly held-out macro F1 for the Bike Sharing model and baseline rules. The random forest leads the weekday-and-hour rule in August and September. The rule leads from October to December.](https://raw.githubusercontent.com/okamasaya/DIY_AutoML/8df22d59dbd5378d52edf87f3aeb3ca55938e86e/docs/dev_community/images/bp-monthly-evaluation.png)

*The overall majority rule always predicts the most frequent label in the training period. Macro F1 gives the busy and regular classes equal weight.*

Across the full period, the weekday-and-hour rule scored **0.656** macro F1 and the random forest scored **0.638**. The random forest caught 80 of the 109 regular cases, and the rule caught 72. The forest missed 29 regular cases versus 37 for the rule. It also labeled 143 busy cases as regular, compared with 115 for the rule. The model scored higher in August and September, and the rule scored higher from October through December. These numbers describe the held-out period. Intervals that account for time dependence remain a research task. In Excel, the simple-rule comparison, monthly breakdown, and monthly breakdown by outcome let you go back to the counts for each period. [Monthly comparison details (Japanese)](https://github.com/okamasaya/DIY_AutoML/blob/64b17f120dad1ead6256d963b779c13630457898/docs/DATA_AUDIT.md#月別の成績と単純ルール)

The charts use the saved evaluation tables. You can also open those worksheets in a charting tool or export them as CSV. For example, use `Model` and `F1` from `SL_CV_Mean` to plot the training-side validation scores. [Charting examples in the development notes](https://github.com/okamasaya/DIY_AutoML/blob/64b17f120dad1ead6256d963b779c13630457898/docs/development-notes/05-business-preview.en.md#take-the-saved-tables-into-a-charting-tool)

## Give the preview a way to open custom charts

One idea is to put PyGWalker inside Streamlit so a user can choose the chart type, axes, and series from the saved tables. In the local test with Python 3.14.7 and PyGWalker 0.5.0.1, an Iris evaluation table produced an HTML file for local viewing. [PyGWalker Streamlit documentation](https://docs.kanaries.net/pygwalker/api-reference/streamlit) · [Prototype log (Japanese)](https://github.com/okamasaya/DIY_AutoML/blob/64b17f120dad1ead6256d963b779c13630457898/docs/THREAD_SUMMARY_20261119.md)

The proposed Windows workflow uses an initial batch file on a PC with Python already installed. The batch would create a charting environment, get the dependencies over the network, and start the viewer. After that, a button in the preview could reuse the environment and open the current evaluation in a local browser. The Streamlit integration, initial setup, and button launch still need to be tested together. [Charting roadmap (Japanese)](https://github.com/okamasaya/DIY_AutoML/blob/64b17f120dad1ead6256d963b779c13630457898/docs/ROADMAP.md#bp向けの詳細グラフto-do)

That wraps up the classification examples. The next article uses the same chronological holdout idea to check predictions of future observations.

*AI assistance was used to adapt the project's development notes and experiment records into this English article.*
