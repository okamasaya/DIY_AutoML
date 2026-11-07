import time
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
from .data import split_indices

METRICS = ("Accuracy", "Precision", "Recall", "F1")

def candidates(seed=42):
    return [("Decision Tree", DecisionTreeClassifier(random_state=seed)),
            ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1)),
            ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=seed)),
            ("SVM", SVC()), ("k-NN", KNeighborsClassifier(n_neighbors=5))]

def pipeline(model):
    return make_pipeline(StandardScaler(), model)

def metrics(y, pred, classes=None):
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=classes, average="macro", zero_division=0)
    return dict(zip(METRICS, map(float, (accuracy_score(y, pred), p, r, f))))

def run(X, y, seed=42):
    train, test = split_indices(y, seed)
    rows, predictions = [], {}
    for name, model in candidates(seed):
        estimator = pipeline(model)
        start = time.perf_counter()
        estimator.fit(X[train], y[train])
        elapsed = time.perf_counter() - start
        pred = estimator.predict(X[test])
        predictions[name] = pred
        rows.append({"Model": name, **metrics(y[test], pred, np.unique(y)), "Train Time (s)": elapsed})
    return pd.DataFrame(rows), {"train": train, "test": test, **{f"prediction_{i}": p for i,p in enumerate(predictions.values())}}
