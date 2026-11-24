"""Track a single scheduled numeric series and save the forecast and its evidence."""

import csv as csv_module
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from .kalman import assess_pilot, filter_series
from .reporting import fail, file_sha256, prepare, save, summary


SCHEMA = 18
CONFIDENCE = 0.95
Z = NormalDist().inv_cdf((1 + CONFIDENCE) / 2)
INPUTS = ("time", "observation", "acceleration", "reference")
METRIC_COLUMNS = ["Stage", "Target", "Method", "Rows", "MAE", "RMSE", "Coverage", "Mean interval width"]


def read_series(path, columns):
    selected = [column for column in columns if column is not None]
    if len(selected) != len(set(selected)):
        raise ValueError("Choose distinct time, observation, acceleration and reference columns")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv_module.reader(stream, strict=True)
        header = next(reader, [])
        if len(header) != len(set(header)) or not set(selected).issubset(header):
            raise ValueError("CSV header must have unique names and contain the selected columns")
        indices = [header.index(column) if column is not None else None for column in columns]
        values = {name: [] for name in INPUTS}
        for row_number, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError(f"Unexpected column count in CSV record {row_number}")
            for name, index in zip(INPUTS, indices):
                value = np.nan
                if index is not None:
                    cell = row[index].strip()
                    if cell or name not in ("observation", "reference"):
                        try:
                            value = float(cell)
                        except ValueError as error:
                            raise ValueError(f"Invalid {name} in CSV record {row_number}") from error
                        if not np.isfinite(value):
                            raise ValueError(f"Nonfinite {name} in CSV record {row_number}")
                elif name == "acceleration":
                    value = 0.0
                values[name].append(value)
    return {name: np.asarray(series, dtype=float) for name, series in values.items()}


def tracking_arrays(inputs, settings):
    model = {name: settings[name] for name in ("observation_sd", "process_noise", "initial_rate", "initial_rate_sd")}
    trace = filter_series(inputs["time"], inputs["observation"], inputs["acceleration"], **model,
                          pilot_rows=settings.get("pilot_rows"), min_rmse_gain=settings.get("min_rmse_gain"))
    return {**inputs, **trace}


def interval(center, variance):
    margin = Z * np.sqrt(variance)
    return center - margin, center + margin


def measure(stage, target, method, truth, prediction, eligible, bounds=None):
    rows = int(np.count_nonzero(eligible))
    error = prediction[eligible] - truth[eligible]
    result = [stage, target, method, rows, np.nan, np.nan, np.nan, np.nan]
    if rows:
        result[4:6] = [np.mean(np.abs(error)), np.sqrt(np.mean(error**2))]
        if bounds is not None:
            lower, upper = (bound[eligible] for bound in bounds)
            result[6:] = [np.mean((lower <= truth[eligible]) & (truth[eligible] <= upper)), np.mean(upper - lower)]
    return result


def tracking_tables(arrays, settings):
    time, observation, acceleration, reference = (arrays[name] for name in INPUTS)
    one_step, estimate, dynamics = (arrays[name][:, 0] for name in ("forecast", "filtered", "dynamics"))
    persistence = arrays["persistence"]
    prior_variance = arrays["forecast_covariance"][:, 0, 0]
    posterior_variance = arrays["filtered_covariance"][:, 0, 0]
    forecast_state_interval = interval(one_step, prior_variance)
    forecast_observation_interval = interval(one_step, prior_variance + settings["observation_sd"] ** 2)
    estimate_interval = interval(estimate, posterior_variance)
    pilot_rows = settings.get("pilot_rows")
    decision = None
    evaluation = np.ones(len(time), dtype=bool)
    selected_method = "Kalman"
    if pilot_rows is not None:
        decision = assess_pilot(observation[:pilot_rows], one_step[:pilot_rows], persistence[:pilot_rows],
                                settings["min_rmse_gain"])
        selected_method = decision["Selected method"]
        evaluation[:pilot_rows] = False
    uses_kalman = selected_method == "Kalman"
    selected_forecast = one_step if uses_kalman else persistence
    has_forecast = np.isfinite(selected_forecast)
    rows = []
    targets = [("Observation", observation, forecast_observation_interval)]
    if settings["reference_column"] is not None:
        targets.append(("Reference state", reference, forecast_state_interval))
    for target, truth, bounds in targets:
        eligible = evaluation & has_forecast & np.isfinite(truth)
        if uses_kalman:
            rows.extend([
                measure("Forecast", target, "Kalman", truth, one_step, eligible, bounds),
                measure("Forecast", target, "Dynamics only", truth, dynamics, eligible),
            ])
        rows.append(measure("Forecast", target, "Last observation", truth, persistence, eligible))
    if uses_kalman and settings["reference_column"] is not None:
        eligible = evaluation & has_forecast & np.isfinite(reference) & np.isfinite(observation)
        rows.extend([
            measure("Filtered", "Reference state", "Kalman", reference, estimate, eligible, estimate_interval),
            measure("Filtered", "Reference state", "Raw observation", reference, observation, eligible),
        ])
    sheets = {"Metrics": pd.DataFrame(rows, columns=METRIC_COLUMNS)}
    if not settings.get("compact_report", False):
        forecast_origin = np.full_like(time, np.nan)
        forecast_origin[1:] = np.where(np.isfinite(persistence[1:]), time[:-1], np.nan)
        track = pd.DataFrame({
            "Time": time, "Forecast origin": forecast_origin, "Observation": observation, "Reference state": reference,
            "Acceleration to next time": acceleration, "Dynamics only": dynamics, "Last observation": persistence,
            "Forecast value": one_step, "Forecast rate": arrays["forecast"][:, 1],
            "Forecast state lower": forecast_state_interval[0], "Forecast state upper": forecast_state_interval[1],
            "Forecast observation lower": forecast_observation_interval[0],
            "Forecast observation upper": forecast_observation_interval[1],
            "Filtered value": estimate, "Filtered rate": arrays["filtered"][:, 1],
            "Filtered state lower": estimate_interval[0], "Filtered state upper": estimate_interval[1],
        })
        if decision is not None:
            scored = evaluation & has_forecast
            track["Period"] = np.where(evaluation, "Evaluation", "Pilot")
            track["Selected method"] = pd.Series(selected_method, index=track.index).where(scored)
            track["Selected forecast"] = np.where(scored, selected_forecast, np.nan)
        sheets["Track"] = track
    sheets["Summary"] = summary([
        ("Model", "Scalar level and rate with white acceleration noise"),
        ("Forecast", "One scheduled step using observations through the previous row"),
        ("Intervals", "Pointwise 95% Gaussian intervals conditional on the model and stated noise"),
        ("Coverage", "Fraction of eligible values inside the corresponding interval"),
        ("Rows", len(time)), ("Missing observations", int(np.count_nonzero(np.isnan(observation)))),
        ("Observation SD", settings["observation_sd"]), ("Process noise intensity", settings["process_noise"]),
        ("Initial rate", settings["initial_rate"]), ("Initial rate SD", settings["initial_rate_sd"]),
    ])
    if decision is not None:
        sheets["Decision"] = pd.DataFrame([{
            "Pilot rows": pilot_rows, "Evaluation starts at": time[pilot_rows],
            **decision, "Kalman forecast steps": int(np.count_nonzero(np.isfinite(one_step))),
        }])
    return sheets


def kalman_track_csv(csv, output, *, time_column, observation_column, observation_sd, process_noise,
                     acceleration_column=None, reference_column=None, initial_rate=0.0, initial_rate_sd=1.0,
                     pilot_rows=None, min_rmse_gain=None, compact_report=False):
    """Evaluate causal one-step forecasts and filtered state estimates from a numeric CSV."""
    path = Path(csv)
    columns = (time_column, observation_column, acceleration_column, reference_column)
    if not time_column or not observation_column:
        raise ValueError("Time and observation columns are required")
    settings = {"csv": str(path), "time_column": time_column, "observation_column": observation_column,
                "acceleration_column": acceleration_column, "reference_column": reference_column,
                "observation_sd": float(observation_sd), "process_noise": float(process_noise),
                "initial_rate": float(initial_rate), "initial_rate_sd": float(initial_rate_sd)}
    if pilot_rows is not None or min_rmse_gain is not None:
        settings.update(pilot_rows=pilot_rows, min_rmse_gain=None if min_rmse_gain is None else float(min_rmse_gain))
    if compact_report:
        settings["compact_report"] = True
    digest = file_sha256(path)
    inputs = read_series(path, columns)
    if file_sha256(path) != digest:
        raise ValueError("The source CSV changed while it was being read")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        arrays = tracking_arrays(inputs, settings)
        sheets = tracking_tables(arrays, settings)
    directory, manifest = prepare(output, SCHEMA, settings)
    try:
        if manifest["input_sha256"] != digest:
            raise ValueError("The source CSV changed after it was read")
        return save(directory, manifest, sheets, arrays)
    except Exception as error:
        fail(directory, manifest, error)
        raise
