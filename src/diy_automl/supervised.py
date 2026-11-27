import time
from collections.abc import Mapping
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from .clustering_subsample import integer
from .data import split_indices

METRICS = ("Accuracy", "Precision", "Recall", "F1")
TREE_MODELS = ("Decision Tree", "Random Forest")

def validate_tree_max_depth(value):
    return None if value is None else integer(value, "tree_max_depth", 1, np.iinfo(np.intp).max)


def tree_depth_settings(tree_max_depth=None, tree_max_depths=None):
    depth = validate_tree_max_depth(tree_max_depth)
    settings = {} if depth is None else {"tree_max_depth": depth}
    if tree_max_depths is not None:
        if not isinstance(tree_max_depths, Mapping) or set(tree_max_depths) - set(TREE_MODELS):
            raise ValueError("tree_max_depths must map Decision Tree or Random Forest to a positive depth")
        overrides = {
            model: integer(tree_max_depths[model], f"{model} max depth", 1, np.iinfo(np.intp).max)
            for model in TREE_MODELS if model in tree_max_depths
        }
        if overrides:
            settings["tree_max_depths"] = overrides
    return settings


def tree_depth_options(settings):
    return {key: settings[key] for key in ("tree_max_depth", "tree_max_depths") if key in settings}


def effective_tree_depths(settings):
    return {**dict.fromkeys(TREE_MODELS, settings.get("tree_max_depth")), **settings.get("tree_max_depths", {})}


def tree_depth_summary(settings):
    depth = settings.get("tree_max_depth")
    rows = [] if depth is None else [("Tree max depth", str(depth))]
    rows.extend((f"{model} max depth", str(limit)) for model, limit in settings.get("tree_max_depths", {}).items())
    return rows


def sl_model_options(settings):
    return {**tree_depth_options(settings), "forest_jobs": settings.get("forest_jobs", 1)}


def candidates(seed=42, *, tree_max_depth=None, tree_max_depths=None, forest_jobs=1):
    depths = effective_tree_depths(tree_depth_settings(tree_max_depth, tree_max_depths))
    decision_tree, random_forest = TREE_MODELS
    return [(decision_tree, DecisionTreeClassifier(random_state=seed, max_depth=depths[decision_tree])),
            (random_forest, RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=forest_jobs,
                                                   max_depth=depths[random_forest])),
            ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=seed)),
            ("SVM", SVC()), ("k-NN", KNeighborsClassifier(n_neighbors=5))]

def pipeline(model):
    return make_pipeline(StandardScaler(), model)

def metrics(y, pred, classes=None, *, sample_weight=None):
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=classes, average="macro",
                                               zero_division=0, sample_weight=sample_weight)
    return dict(zip(METRICS, map(float, (accuracy_score(y, pred, sample_weight=sample_weight), p, r, f))))

def run(X, y, seed=42, *, tree_max_depth=None, tree_max_depths=None, forest_jobs=1):
    train, test = split_indices(y, seed)
    rows, predictions = [], {}
    for name, model in candidates(seed, tree_max_depth=tree_max_depth, tree_max_depths=tree_max_depths,
                                  forest_jobs=forest_jobs):
        estimator = pipeline(model)
        start = time.perf_counter()
        estimator.fit(X[train], y[train])
        elapsed = time.perf_counter() - start
        pred = estimator.predict(X[test])
        predictions[name] = pred
        rows.append({"Model": name, **metrics(y[test], pred, np.unique(y)), "Train Time (s)": elapsed})
    return pd.DataFrame(rows), {"train": train, "test": test, **{f"prediction_{i}": p for i,p in enumerate(predictions.values())}}
