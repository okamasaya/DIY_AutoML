"""Hold out recent observation days and validate on past-to-future windows."""

from datetime import date

import numpy as np


def chronological_partitions(days):
    ordered = np.argsort(days, kind="stable")
    dates, counts = np.unique(days, return_counts=True)
    holdout_days = (len(dates) + 4) // 5
    if len(dates) - holdout_days < 6:
        raise ValueError("Chronological selection needs at least six training dates after reserving the latest 20% of dates")

    training_dates = dates[:-holdout_days]
    cut = int(counts[:-holdout_days].sum())
    train, test = ordered[:cut], ordered[cut:]
    windows = np.array_split(training_dates, 6)
    edges = np.searchsorted(days[train], [window[0] for window in windows[1:]])
    folds = [(np.arange(start), np.arange(start, end))
             for start, end in zip(edges, [*edges[1:], len(train)])]
    return train, test, folds


def period(days):
    return {"start": date.fromordinal(int(days.min())).isoformat(),
            "end": date.fromordinal(int(days.max())).isoformat(),
            "days": len(np.unique(days)), "rows": len(days)}


def chronology(days, clock, train, test, folds):
    return {
        "mode": "chronological", "clock": clock,
        "training": period(days[train]), "test": period(days[test]),
        "folds": [{"training": period(days[train[a]]), "validation": period(days[train[b]])}
                  for a, b in folds],
    }


def require_class_coverage(labels, train, test, folds):
    classes = len(np.unique(labels))
    first_training = train[folds[0][0]]
    if len(np.unique(labels[first_training])) != classes:
        raise ValueError("The earliest training window must contain every class")
    if len(first_training) < 5:
        raise ValueError("The earliest training window needs at least five rows for k-NN")
    for index, (_, validation) in enumerate(folds, start=1):
        if len(np.unique(labels[train[validation]])) != classes:
            raise ValueError(f"Chronological validation window {index} must contain every class")
    if len(np.unique(labels[test])) != classes:
        raise ValueError("The chronological holdout period must contain every class")
