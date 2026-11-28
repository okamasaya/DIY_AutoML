"""Public Python API for UC_AutoMaiL."""

import diy_automl as _implementation

__version__ = _implementation.__version__
__all__ = _implementation.__all__


def __getattr__(name):
    if not name.startswith("_"):
        try:
            return getattr(_implementation, name)
        except AttributeError:
            pass
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
