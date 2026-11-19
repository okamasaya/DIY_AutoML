"""Holdout comparisons for the selected classifier and training-only calendar rules."""

from collections import Counter, defaultdict
from datetime import date

import numpy as np
import pandas as pd

from .chronological_selection import period
from .supervised import metrics


METHODS = ("Selected model", "Calendar rule", "Overall majority")
COMPARISON = "SL_Test_Comparison"
BY_MONTH = "SL_Test_By_Month"
BY_MONTH_CLASS = "SL_Test_By_Month_Class"
COMPARISON_SHEETS = {COMPARISON, BY_MONTH, BY_MONTH_CLASS}


def majority(counts):
    return min(counts, key=lambda label: (-counts[label], label))


def calendar_predictions(train_days, train_hours, train_truth, test_days, test_hours):
    weekdays, weekday_hours = defaultdict(Counter), defaultdict(Counter)
    for weekday, hour, label in zip((train_days - 1) % 7, train_hours, train_truth):
        weekdays[weekday][label] += 1
        if hour >= 0:
            weekday_hours[weekday, hour][label] += 1
    fallback = majority(Counter(train_truth))
    by_weekday = {day: majority(counts) for day, counts in weekdays.items()}
    by_hour = {slot: majority(counts) for slot, counts in weekday_hours.items()}

    predictions = np.empty(len(test_days), dtype=train_truth.dtype)
    sources = Counter()
    for i, (weekday, hour) in enumerate(zip((test_days - 1) % 7, test_hours)):
        if (weekday, hour) in by_hour:
            label, source = by_hour[weekday, hour], "weekday_hour"
        elif weekday in by_weekday:
            label, source = by_weekday[weekday], "weekday"
        else:
            label, source = fallback, "overall"
        predictions[i] = label
        sources[source] += 1
    return predictions, np.full_like(predictions, fallback), {
        "uses_hours": bool(by_hour),
        "source_rows": {key: sources[key] for key in ("weekday_hour", "weekday", "overall")},
    }


def comparison_rows(truth, predictions, classes):
    return [{"Method": method, "Rows": len(truth), "Correct": int(np.count_nonzero(truth == pred)),
             **metrics(truth, pred, classes)} for method, pred in zip(METHODS, predictions)]


def class_rows(truth, predictions, classes):
    support = np.bincount(truth, minlength=len(classes))
    for method, pred in zip(METHODS, predictions):
        called = np.bincount(pred, minlength=len(classes))
        matches = np.bincount(truth[truth == pred], minlength=len(classes))
        for label, actual, predicted, correct in zip(classes, support, called, matches):
            yield {"Method": method, "Class index": label, "Support": actual, "Correct": correct,
                   "Missed": actual - correct, "False positives": predicted - correct,
                   "Recall": correct / actual if actual else np.nan}


def calendar_comparison(arrays, class_count):
    train, test = arrays["train"], arrays["test"]
    days, hours = arrays["observation_days"], arrays["observation_hours"]
    calendar, overall, rule = calendar_predictions(days[train], hours[train], arrays["train_truth"],
                                                   days[test], hours[test])
    predictions = np.stack([arrays["predictions"], calendar, overall])
    truth, classes, test_days = arrays["truth"], np.arange(class_count), days[test]
    whole = pd.DataFrame(comparison_rows(truth, predictions, classes))
    monthly, monthly_classes = [], []
    unique_days, day_index = np.unique(test_days, return_inverse=True)
    month_of_day = np.array([date.fromordinal(int(day)).isoformat()[:7] for day in unique_days])
    months, counts = np.unique(month_of_day[day_index], return_counts=True)
    # Chronological holdout indices already order rows by date.
    for month, rows in zip(months, np.split(np.arange(len(test)), np.cumsum(counts)[:-1])):
        window = period(test_days[rows])
        dates = {"Month": month, "First date": window["start"], "Last date": window["end"],
                 "Observed days": window["days"]}
        pred = predictions[:, rows]
        monthly.extend({**dates, **row} for row in comparison_rows(truth[rows], pred, classes))
        monthly_classes.extend({"Month": month, **row} for row in class_rows(truth[rows], pred, classes))
    return {COMPARISON: whole, BY_MONTH: pd.DataFrame(monthly), BY_MONTH_CLASS: pd.DataFrame(monthly_classes)}, {
        "calendar_predictions": calendar, "majority_predictions": overall,
    }, rule
