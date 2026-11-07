import csv as csv_module
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    feature_names: list
    classes: list
    source: str

def load_csv(csv, target):
    with Path(csv).open(encoding="utf-8-sig", newline="") as stream:
        header = next(csv_module.reader(stream), [])
    if len(header) != len(set(header)):
        raise ValueError("Duplicate CSV columns are not supported")
    frame = pd.read_csv(csv)
    if not frame.columns.is_unique or target not in frame:
        raise ValueError("A unique target column is required")
    features = frame.drop(columns=target)
    if features.shape[1] == 0 or frame.isna().any().any():
        raise ValueError("At least one feature and no missing values are required")
    if any(not pd.api.types.is_numeric_dtype(t) for t in features.dtypes):
        raise ValueError("Only numeric features are supported")
    X = features.to_numpy(dtype=float)
    if not np.isfinite(X).all():
        raise ValueError("Features must be finite")
    encoder = LabelEncoder()
    y = encoder.fit_transform(frame[target])
    if len(encoder.classes_) < 2:
        raise ValueError("At least two target classes are required")
    return Dataset(X, y, list(features.columns), encoder.classes_.tolist(), str(Path(csv).resolve()))

def split_indices(y, seed=42):
    train, test = train_test_split(np.arange(len(y)), test_size=.2, random_state=seed, stratify=y)
    return train, test

def dl_indices(y, split_seed=42):
    train_pool, test = split_indices(y, split_seed)
    train, validation = train_test_split(train_pool, test_size=.2, random_state=split_seed, stratify=y[train_pool])
    return train, validation, test

def scale_train(X, train):
    scaler = StandardScaler().fit(X[train])
    return scaler.transform(X), scaler
