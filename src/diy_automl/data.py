import csv as csv_module
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from .data_profile import MISSING, parse_timestamp

@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    feature_names: list
    classes: list
    source: str
    groups: np.ndarray | None = None
    observation_days: np.ndarray | None = None
    clock: str | None = None
    observation_hours: np.ndarray | None = None

def load_csv(csv, target, *, target_as_text=False, group_column=None, time_column=None):
    with Path(csv).open(encoding="utf-8-sig", newline="") as stream:
        header = next(csv_module.reader(stream), [])
    if len(header) != len(set(header)):
        raise ValueError("Duplicate CSV columns are not supported")
    types = {target: str} if target_as_text else {}
    if group_column is not None:
        if group_column == target or group_column not in header:
            raise ValueError("Choose an existing group column distinct from the target")
        types[group_column] = str
    if time_column is not None:
        if time_column in (target, group_column) or time_column not in header:
            raise ValueError("Choose an existing timestamp column distinct from the target and identifier")
        types[time_column] = str
    frame = pd.read_csv(csv, engine="python", dtype=types) if types else pd.read_csv(csv)
    if not frame.columns.is_unique or target not in frame:
        raise ValueError("A unique target column is required")
    excluded = [target]
    groups = None
    if group_column is not None:
        identifiers = frame[group_column]
        if identifiers.isna().any() or identifiers.str.strip().isin(MISSING).any():
            raise ValueError("Group identifiers cannot be missing")
        groups, _ = pd.factorize(identifiers, sort=False)
        excluded.append(group_column)
    observation_days, observation_hours, clock = None, None, None
    if time_column is not None:
        observation_days, observation_hours, clock = observation_dates(frame[time_column])
        excluded.append(time_column)
    features = frame.drop(columns=excluded)
    if features.shape[1] == 0 or frame.isna().any().any():
        raise ValueError("At least one feature and no missing values are required")
    if any(not pd.api.types.is_numeric_dtype(t) for t in features.dtypes):
        raise ValueError("Only numeric features are supported")
    X = features.to_numpy(dtype=float)
    if not np.isfinite(X).all():
        raise ValueError("Features must be finite")
    encoder = LabelEncoder()
    labels = frame[target]
    if target_as_text and labels.str.strip().isin(MISSING).any():
        raise ValueError("No missing target labels are allowed")
    y = encoder.fit_transform(labels)
    if len(encoder.classes_) < 2:
        raise ValueError("At least two target classes are required")
    return Dataset(X, y, list(features.columns), encoder.classes_.tolist(), str(Path(csv).resolve()),
                   groups=groups, observation_days=observation_days, clock=clock, observation_hours=observation_hours)


def observation_dates(values):
    days, hours, zones = [], [], set()
    for row, value in enumerate(values, start=1):
        if pd.isna(value) or value.strip() in MISSING:
            raise ValueError(f"Missing timestamp in data row {row}")
        try:
            point = date.fromisoformat(value.strip())
            hour, zone = -1, "local_unspecified"
        except ValueError:
            try:
                point, zone = parse_timestamp(value)
            except (ValueError, OverflowError) as exc:
                raise ValueError(f"Expected an ISO 8601 timestamp in data row {row}") from exc
            hour = point.hour
        days.append(point.toordinal())
        hours.append(hour)
        zones.add(zone)
    if len(zones) != 1:
        raise ValueError("Supply observations using either offset-aware or local timestamps consistently")
    return np.asarray(days, dtype=np.int64), np.asarray(hours, dtype=np.int8), zones.pop()

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
