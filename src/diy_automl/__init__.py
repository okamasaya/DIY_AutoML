"""New implementation; not recovered historical source."""
from importlib import import_module as _import_module
from typing import TYPE_CHECKING as _TYPE_CHECKING

__version__ = "0.4.0a7"
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
    from .extended_kalman import extended_kalman_filter as extended_kalman_filter
    from .unscented_kalman import unscented_kalman_filter as unscented_kalman_filter
    from .particle_filter import bootstrap_particle_filter as bootstrap_particle_filter

# Explicit use of a state estimator loads its implementation on demand.
_FILTER_EXPORTS = {
    "kalman_track_csv": ".kalman_workflow",
    "extended_kalman_filter": ".extended_kalman",
    "unscented_kalman_filter": ".unscented_kalman",
    "bootstrap_particle_filter": ".particle_filter",
}
__all__ = [name for name in globals() if not name.startswith("_")] + list(_FILTER_EXPORTS)


def __getattr__(name):
    module = _FILTER_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    function = getattr(_import_module(module, __name__), name)
    globals()[name] = function
    return function


def __dir__():
    return sorted(set(globals()) | _FILTER_EXPORTS.keys())
