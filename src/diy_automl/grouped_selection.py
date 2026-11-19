"""Group-disjoint partitions for supervised model selection."""

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def identical_input_groups(features):
    """Use exact equality of the numeric features the estimators see."""
    _, groups = np.unique(features, axis=0, return_inverse=True)
    return groups.reshape(-1)


def group_folds(labels, groups, seed):
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    return splitter.split(np.zeros(len(labels)), labels, groups)


def grouped_partitions(labels, groups, seed):
    """Find a held-out fold that supports the same five-fold selection on its complement."""
    classes = np.unique(labels)
    if any(len(np.unique(groups[labels == label])) < 6 for label in classes):
        raise ValueError("Grouped five-fold selection and holdout need each class in at least six groups")

    for fold, (train, test) in enumerate(group_folds(labels, groups, seed), start=1):
        if len(np.unique(labels[test])) != len(classes) or len(np.unique(groups[test])) < 2:
            continue
        train_labels, train_groups = labels[train], groups[train]
        if any(len(np.unique(train_groups[train_labels == label])) < 5 for label in classes):
            continue
        folds = list(group_folds(train_labels, train_groups, seed))
        if all(len(np.unique(train_labels[b])) == len(classes) for _, b in folds):
            return train, test, folds, fold
    raise ValueError("Could not keep every class in the holdout and all five validation folds by group")
