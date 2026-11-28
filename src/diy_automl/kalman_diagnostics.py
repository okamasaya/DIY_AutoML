"""Offline consistency checks for Gaussian state-space filter results."""

from numbers import Integral

import numpy as np
from scipy.stats import chi2, norm

from .state_space import check_covariance, finite_array


def _diagnostic_inputs(values, mean, covariance, *, positive):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not values.size or np.isinf(values).any():
        raise ValueError("Diagnostic values must be a nonempty (rows, dimensions) array of finite values or NaN")
    rows, dimensions = values.shape
    mean = finite_array(mean, values.shape, "Filter mean")
    covariance = check_covariance(covariance, dimensions, "Filter covariance", steps=rows, positive=positive)
    return values, mean, covariance


def _quadratic_scores(residuals, covariance, confidence, *, allow_singular=False):
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    scores = np.full(len(residuals), np.nan)
    dimensions = np.zeros(len(residuals), dtype=int)
    for index, row in enumerate(residuals):
        available = np.flatnonzero(~np.isnan(row))
        if not available.size:
            continue
        try:
            factor = np.linalg.cholesky(covariance[index][np.ix_(available, available)])
        except np.linalg.LinAlgError:
            if allow_singular:
                continue
            raise
        whitened = np.linalg.solve(factor, row[available])
        scores[index] = whitened @ whitened
        dimensions[index] = available.size

    tail = (1 - confidence) / 2
    intervals = np.full((len(residuals), 2), np.nan)
    scored = dimensions > 0
    intervals[scored] = chi2.ppf([tail, 1 - tail], dimensions[scored, None])
    return scores, dimensions, intervals


def _standardize(residuals, covariance):
    deviation = np.sqrt(np.maximum(np.diagonal(covariance, axis1=1, axis2=2), 0))
    return np.divide(residuals, deviation, out=np.full_like(residuals, np.nan), where=deviation > 0)


def _innovation_autocorrelation(normalized, lags):
    rows, channels = normalized.shape
    autocorrelation = np.full((len(lags), channels), np.nan)
    pairs = np.zeros((len(lags), channels), dtype=int)
    ljung_box = np.full(channels, np.nan)
    status = ["no_lags"] * channels
    if not len(lags):
        return autocorrelation, pairs, ljung_box, tuple(status)
    for channel, series in enumerate(normalized.T):
        observed = ~np.isnan(series)
        values = series[observed]
        if not values.size:
            status[channel] = "missing_observations"
            continue
        # Center relative to one sample first to keep a large offset out of the mean.
        shifted = values - values[0]
        centered = np.zeros(rows)
        centered[observed] = shifted - shifted.mean()
        variation = centered @ centered
        for index, lag in enumerate(lags):
            pairs[index, channel] = np.count_nonzero(observed[:-lag] & observed[lag:])
            if variation > 0 and pairs[index, channel]:
                autocorrelation[index, channel] = centered[:-lag] @ centered[lag:] / variation
        if values.size < rows:
            status[channel] = "missing_observations"
        elif variation == 0:
            status[channel] = "constant_series"
        elif np.isfinite(variation):
            ljung_box[channel] = rows * (rows + 2) * np.sum(autocorrelation[:, channel] ** 2 / (rows - lags))
            status[channel] = "ok"
        else:
            status[channel] = "nonfinite_variation"

    return autocorrelation, pairs, ljung_box, tuple(status)


def filter_diagnostics(result, observations, *, max_lag=10, confidence=0.95):
    """Measure innovation magnitude and serial correlation in an EKF or UKF result.

    Pass the same observation rows, with NaN for missing entries. The normalized
    innovation is standardized using each channel's predictive variance. NIS uses
    the full covariance of the available channels at each row. The per-row chi-square
    reference interval uses their count as degrees of freedom.

    Autocorrelations use the original row lags and the available pairs at each lag.
    Ljung–Box statistics and approximate p-values are computed per complete,
    nonconstant channel. Rows are assumed to be evenly spaced for time-lag analysis.
    The Ljung–Box status is per channel. Complete, nonconstant channels also get
    a pointwise white-noise reference band of ±z/sqrt(rows) at ``confidence``
    and the lags outside that band. A NaN limit marks an unavailable band.
    """
    if not isinstance(max_lag, Integral) or max_lag < 0:
        raise ValueError("max_lag must be a nonnegative integer")
    observed, predicted, covariance = _diagnostic_inputs(
        observations, result["predicted_observation"], result["predicted_observation_covariance"], positive=True,
    )
    innovation = observed - predicted
    normalized = _standardize(innovation, covariance)
    nis, dimensions, intervals = _quadratic_scores(innovation, covariance, confidence)
    lags = np.arange(1, min(max_lag, len(observed) - 1) + 1)
    autocorrelation, pairs, ljung_box, status = _innovation_autocorrelation(normalized, lags)
    reference_limit = norm.isf((1 - confidence) / 2) / np.sqrt(len(observed))
    limits = np.where(np.isfinite(ljung_box), reference_limit, np.nan)
    total_dimensions = dimensions.sum()
    return {
        "innovation": innovation, "normalized_innovation": normalized,
        "nis": nis, "observed_dimensions": dimensions, "nis_reference_interval": intervals,
        "mean_nis_per_dimension": np.nansum(nis) / total_dimensions if total_dimensions else np.nan,
        "lags": lags, "autocorrelation": autocorrelation, "autocorrelation_pairs": pairs,
        "autocorrelation_reference_limit": limits,
        "autocorrelation_lags_outside_reference": tuple(
            lags[np.abs(series) > limit] for series, limit in zip(autocorrelation.T, limits, strict=True)
        ),
        "ljung_box_statistic": ljung_box, "ljung_box_pvalue": chi2.sf(ljung_box, len(lags)),
        "ljung_box_status": status,
    }


def evaluate_filter_consistency(truth, result, *, confidence=0.95):
    """Compare posterior estimates and their covariance with known state rows.

    Pulls are (estimate - truth) divided by the posterior marginal standard deviation.
    A missing truth component or zero variance produces a NaN pull. Pull summaries
    are per state, with sample standard deviation. NEES uses the covariance submatrix
    for known components. A singular submatrix or entirely missing row produces NaN
    NEES and zero degrees of freedom. Chi-square intervals use the scored dimensions.
    """
    truth, filtered, covariance = _diagnostic_inputs(
        truth, result["filtered"], result["filtered_covariance"], positive=False,
    )
    error = filtered - truth
    pull = _standardize(error, covariance)
    nees, dimensions, intervals = _quadratic_scores(error, covariance, confidence, allow_singular=True)
    pull_mean = np.full(pull.shape[1], np.nan)
    pull_std = np.full_like(pull_mean, np.nan)
    pull_rms = np.full_like(pull_mean, np.nan)
    for component, series in enumerate(pull.T):
        valid = series[~np.isnan(series)]
        if valid.size:
            pull_mean[component] = valid.mean()
            pull_rms[component] = np.sqrt(np.mean(valid**2))
        if valid.size > 1:
            pull_std[component] = valid.std(ddof=1)

    return {
        "error": error, "pull": pull, "pull_mean": pull_mean, "pull_std": pull_std, "pull_rms": pull_rms,
        "nees": nees, "nees_dimensions": dimensions, "nees_reference_interval": intervals,
    }
