"""Sampling uncertainty of fixed classification predictions, not training variability."""
from numbers import Real

import numpy as np
import pandas as pd

from .clustering_subsample import integer
from .supervised import METRICS

MAX_ROWS = 10000
MAX_CLASSES = 50
MAX_EVALUATIONS = 20
MAX_DRAWS = 10_000_000
MAX_SCORE_VALUES = 5_000_000
CLASS_METRICS = ('Precision', 'Recall', 'F1')
INTERVAL_COLUMNS = ['Metric', 'Estimate', 'Bootstrap Mean', 'Bootstrap SE',
                    'Lower', 'Upper', 'Interval Width', 'Confidence', 'Replicates',
                    'Zero Width', 'Status']


def configuration(repeats=1000, confidence=.95, seed=43, resampling='stratified'):
    repeats = integer(repeats, 'bootstrap_repeats', 2, 10000)
    seed = integer(seed, 'bootstrap_seed')
    if isinstance(confidence, bool) or not isinstance(confidence, Real) or not 0 < confidence < 1:
        raise ValueError('confidence must be strictly between 0 and 1')
    if resampling not in ('stratified', 'ordinary'):
        raise ValueError('resampling must be stratified or ordinary')
    return repeats, float(confidence), seed, resampling


def observations(truth, predictions, classes):
    y, pred, labels = map(np.asarray, (truth, predictions, classes))
    if (y.ndim != 1 or not 1 <= len(y) <= MAX_ROWS or pred.shape != y.shape
            or labels.ndim != 1 or not 2 <= len(labels) <= MAX_CLASSES
            or not np.array_equal(labels, np.arange(len(labels)))
            or not all(np.issubdtype(a.dtype, np.integer) for a in (y, pred, labels))
            or not np.isin(y, labels).all() or not np.isin(pred, labels).all()):
        raise ValueError('Invalid encoded classification observations or size limit exceeded')
    return y.astype(np.int64), pred.astype(np.int64), labels.astype(np.int64)


def resample_indices(truth, repeats=1000, seed=43, resampling='stratified'):
    repeats, _, seed, resampling = configuration(repeats, .95, seed, resampling)
    y = np.asarray(truth)
    if y.ndim != 1 or not 1 <= len(y) <= MAX_ROWS or repeats * len(y) > MAX_DRAWS:
        raise ValueError('Bootstrap row draw limit exceeded or invalid truth')
    rng = np.random.default_rng(seed)
    indices = np.empty((repeats, len(y)), dtype=np.int32)
    groups = [np.flatnonzero(y == c) for c in np.unique(y)]
    for b in range(repeats):
        indices[b] = (np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
                      if resampling == 'stratified' else rng.choice(len(y), len(y), replace=True))
    return indices


def scores_from_confusion(matrix):
    matrix = np.asarray(matrix)
    support = matrix.sum(axis=-1)
    predicted = matrix.sum(axis=-2)
    tp = np.diagonal(matrix, axis1=-2, axis2=-1)
    def divide(a, b):
        return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)
    precision, recall = divide(tp, predicted), divide(tp, support)
    f1 = divide(2 * tp, support + predicted)
    overall = np.stack([divide(tp.sum(axis=-1), support.sum(axis=-1)),
                        precision.mean(axis=-1), recall.mean(axis=-1), f1.mean(axis=-1)], axis=-1)
    per_class = np.stack([precision, recall, f1], axis=-1)
    return overall, per_class


def score_samples(truth, predictions, classes, indices):
    y, pred, labels = observations(truth, predictions, classes)
    indices = np.asarray(indices)
    if (indices.ndim != 2 or indices.shape[1] != len(y) or len(indices) < 2
            or not np.issubdtype(indices.dtype, np.integer)
            or (indices < 0).any() or (indices >= len(y)).any()):
        raise ValueError('Invalid bootstrap indices')
    c = len(labels)
    encoded = y * c + pred
    point_matrix = np.bincount(encoded, minlength=c*c).reshape(c, c)
    point, class_point = scores_from_confusion(point_matrix)
    samples = np.empty((len(indices), len(METRICS)))
    class_samples = np.empty((len(indices), c, len(CLASS_METRICS)))
    for b, draw in enumerate(indices):
        weights = np.bincount(draw, minlength=len(y))
        matrix = np.bincount(encoded, weights=weights, minlength=c*c).reshape(c, c)
        samples[b], class_samples[b] = scores_from_confusion(matrix)
    return point, class_point, samples, class_samples, point_matrix


def interval_rows(estimates, samples, names, confidence, absent=False):
    alpha = (1 - confidence) / 2
    bounds = np.quantile(samples, [alpha, 1-alpha], axis=0, method='linear')
    # SD of bootstrap estimates approximates the estimator's SE. Never divide by sqrt(B).
    se = samples.std(axis=0, ddof=1)
    means = samples.mean(axis=0)
    return pd.DataFrame([
        {'Metric': name, 'Estimate': float(estimates[j]), 'Bootstrap Mean': float(means[j]),
         'Bootstrap SE': float(se[j]), 'Lower': float(bounds[0,j]), 'Upper': float(bounds[1,j]),
         'Interval Width': float(bounds[1,j]-bounds[0,j]), 'Confidence': confidence,
         'Replicates': len(samples), 'Zero Width': bool(bounds[0,j] == bounds[1,j]),
         'Status': 'class_absent_in_test' if absent else 'conditional_row_bootstrap'}
        for j, name in enumerate(names)], columns=INTERVAL_COLUMNS)


def evaluate_metric_uncertainty(truth, predictions, classes, repeats=1000,
                                confidence=.95, seed=43, resampling='stratified'):
    repeats, confidence, seed, resampling = configuration(repeats, confidence, seed, resampling)
    y, pred, labels = observations(truth, predictions, classes)
    if repeats * (4+3*len(labels)) > MAX_SCORE_VALUES:
        raise ValueError('Bootstrap score storage limit exceeded')
    indices = resample_indices(y, repeats, seed, resampling)
    point, class_point, samples, class_samples, matrix = score_samples(y, pred, labels, indices)
    overall = interval_rows(point, samples, METRICS, confidence)
    class_rows = []
    for c in labels:
        frame = interval_rows(class_point[c], class_samples[:,c,:], CLASS_METRICS,
                              confidence, absent=matrix[c].sum() == 0)
        frame.insert(0, 'Encoded Class', int(c))
        frame.insert(1, 'Test Support', int(matrix[c].sum()))
        frame.insert(2, 'Predicted Support', int(matrix[:,c].sum()))
        frame.insert(3, 'TP', int(matrix[c,c]))
        frame.insert(4, 'FP', int(matrix[:,c].sum()-matrix[c,c]))
        frame.insert(5, 'FN', int(matrix[c].sum()-matrix[c,c]))
        class_rows.append(frame)
    return overall, pd.concat(class_rows, ignore_index=True), {
        'truth': y, 'predictions': pred, 'indices': indices,
        'samples': samples, 'class_samples': class_samples, 'confusion': matrix,
    }
