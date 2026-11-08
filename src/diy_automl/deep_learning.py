"""Shared DL preparation and checked dispatch to the TensorFlow reference adapter."""
import numpy as np

from .data import dl_indices, scale_train
from .dl_contract import (HISTORY_COLUMNS, PreparedDL, TrainingFailure, integer,
                          preparation_fingerprint, require_unchanged_preparation,
                          validate_result)
from .tensorflow_backend import MODEL_CONFIG

def prepare_data(X, y, split_seed=42):
    split_seed = integer(split_seed, 'split_seed', maximum=2**32-1)
    train, validation, test = dl_indices(y, split_seed)
    scaled, scaler = scale_train(X, train)
    values = [scaled, np.array(y, copy=True), train, validation, test,
              scaler.mean_, scaler.scale_]
    for value in values:
        value.setflags(write=False)
    return PreparedDL(*values, split_seed)


def fit_seed(prepared, training_seed=42, epochs=100, batch_size=16):
    from .tensorflow_backend import fit_seed as fit_tensorflow
    fingerprint=preparation_fingerprint(prepared)
    try:
        result = fit_tensorflow(prepared, training_seed, epochs, batch_size)
    except Exception:
        require_unchanged_preparation(prepared,fingerprint)
        raise
    require_unchanged_preparation(prepared,fingerprint)
    validate_result(prepared, result, training_seed, epochs, batch_size)
    return result


def run(X, y, seed=42, split_seed=None, training_seed=None, epochs=100, batch_size=16):
    # Legacy seed supplies both defaults; explicit training_seed never changes indices.
    split_seed = seed if split_seed is None else split_seed
    training_seed = seed if training_seed is None else training_seed
    return fit_seed(prepare_data(X, y, split_seed), training_seed, epochs, batch_size)
