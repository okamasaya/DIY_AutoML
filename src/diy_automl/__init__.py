"""New implementation; not recovered historical source."""
from importlib import import_module as _import_module
from typing import TYPE_CHECKING as _TYPE_CHECKING

__version__ = "0.4.1a1"
from .workflow import run_csv
from .selection_workflow import select_csv
from .clustering_repeat_workflow import repeat_clustering_csv
from .deep_learning_repeat_workflow import repeat_deep_learning_csv
from .clustering_subsample_workflow import subsample_clustering_csv
from .tuning_workflow import tune_csv
from .excel_report import render_excel_report
from .uncertainty_workflow import evaluate_uncertainty
from .metric_uncertainty import evaluate_metric_uncertainty
from .sl_ranking import rank_sl_csv
from .sl_uncertainty_audit import audit_sl_uncertainty
from .annealing_workflow import annealing_check_csv
from .data_audit import audit_data_csv, prepare_bp_csv
from .data_extract import extract_data

if _TYPE_CHECKING:
    from .kalman_workflow import kalman_track_csv as kalman_track_csv
    from .extended_kalman import ExtendedKalmanFilter as ExtendedKalmanFilter
    from .extended_kalman import extended_kalman_filter as extended_kalman_filter
    from .unscented_kalman import unscented_kalman_filter as unscented_kalman_filter
    from .particle_filter import bootstrap_particle_filter as bootstrap_particle_filter
    from .graph_classification import classify_graph_nodes as classify_graph_nodes
    from .kalman_diagnostics import evaluate_filter_consistency as evaluate_filter_consistency
    from .kalman_diagnostics import filter_diagnostics as filter_diagnostics

# Optional workflows load their implementation on first use.
_ON_DEMAND_EXPORTS = {
    "kalman_track_csv": ".kalman_workflow",
    "ExtendedKalmanFilter": ".extended_kalman",
    "extended_kalman_filter": ".extended_kalman",
    "unscented_kalman_filter": ".unscented_kalman",
    "bootstrap_particle_filter": ".particle_filter",
    "classify_graph_nodes": ".graph_classification",
    "evaluate_filter_consistency": ".kalman_diagnostics",
    "filter_diagnostics": ".kalman_diagnostics",
}
__all__ = [
    "run_csv", "select_csv", "repeat_clustering_csv", "repeat_deep_learning_csv",
    "subsample_clustering_csv", "tune_csv", "render_excel_report", "evaluate_uncertainty",
    "evaluate_metric_uncertainty", "rank_sl_csv", "audit_sl_uncertainty", "annealing_check_csv",
    "audit_data_csv", "prepare_bp_csv", "extract_data", *_ON_DEMAND_EXPORTS,
]


def __getattr__(name):
    module = _ON_DEMAND_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    function = getattr(_import_module(module, __name__), name)
    globals()[name] = function
    return function


def __dir__():
    return sorted(set(globals()) | _ON_DEMAND_EXPORTS.keys())
