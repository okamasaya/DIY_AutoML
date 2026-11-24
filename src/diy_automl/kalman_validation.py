"""Recompute the saved filter trace and evaluation tables from their recorded inputs."""

import numpy as np

from .kalman_workflow import INPUTS, tracking_arrays, tracking_tables
from .ul_subsample_validation import same_table


def validate_tracking(manifest, arrays, sheets):
    if not set(INPUTS).issubset(arrays.files):
        raise ValueError("Missing tracking inputs")
    inputs = {name: arrays[name] for name in INPUTS}
    settings = manifest["settings"]
    reference = inputs["reference"]
    if reference.shape != inputs["time"].shape or np.isinf(reference).any():
        raise ValueError("Invalid state reference")
    if settings["reference_column"] is None and not np.isnan(reference).all():
        raise ValueError("Unexpected state reference")
    if settings["acceleration_column"] is None and np.any(inputs["acceleration"] != 0):
        raise ValueError("Unexpected acceleration")
    expected = tracking_arrays(inputs, settings)
    if set(arrays.files) != set(expected):
        raise ValueError("Tracking arrays differ")
    for name, values in expected.items():
        if arrays[name].shape != values.shape or not np.allclose(arrays[name], values, atol=1e-12, rtol=1e-12,
                                                                 equal_nan=True):
            raise ValueError(f"Tracking array differs: {name}")
    tables = tracking_tables(expected, settings)
    if set(sheets) != set(tables):
        raise ValueError("Tracking sheets differ")
    for name, table in tables.items():
        same_table(sheets[name], table, f"Tracking sheet differs: {name}")
