"""CPU settings for supervised classification and its replay workflows."""

from contextlib import contextmanager
from numbers import Integral
from threading import RLock

from threadpoolctl import threadpool_limits

from .clustering_subsample import integer


_MAX_CPU_COUNT = 2**31 - 1
_THREAD_LIMIT_LOCK = RLock()


def sl_compute_settings(cpu_threads=None, forest_jobs=None):
    settings = {}
    if cpu_threads is not None:
        settings["cpu_threads"] = integer(cpu_threads, "cpu_threads", 1, _MAX_CPU_COUNT)
    if forest_jobs is not None:
        if isinstance(forest_jobs, Integral) and forest_jobs == -1:
            settings["forest_jobs"] = -1
        else:
            settings["forest_jobs"] = integer(forest_jobs, "forest_jobs", 1, _MAX_CPU_COUNT)
    return settings


def sl_compute_options(settings):
    return {name: settings[name] for name in ("cpu_threads", "forest_jobs") if name in settings}


def sl_compute_summary(settings):
    titles = {"cpu_threads": "Native CPU threads", "forest_jobs": "Random Forest jobs"}
    return [(title, str(settings[name])) for name, title in titles.items() if name in settings]


@contextmanager
def limit_sl_threads(settings):
    limit = settings.get("cpu_threads")
    if limit is None:
        yield
    else:
        # Some native thread pools are process-wide. Serialize changes to their limits.
        with _THREAD_LIMIT_LOCK, threadpool_limits(limits=limit):
            yield
