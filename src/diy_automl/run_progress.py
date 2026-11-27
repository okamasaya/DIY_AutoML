"""Opt-in progress for supervised classification workflows."""

from contextlib import suppress
import json
import sys
from time import perf_counter

import pandas as pd

from .reporting import file_sha256


MODES = ("off", "console", "excel", "both")
PROGRESS_LOG = "progress.jsonl"
PROGRESS_SHEET = "Progress"
FIELDS = ("elapsed_seconds", "stage", "model", "candidate", "candidates", "fold", "folds", "metric", "score")


def progress_mode(value):
    if value not in MODES:
        raise ValueError("Progress must be off, console, excel or both")
    return value


class RunProgress:
    def __init__(self, mode, directory):
        self.path = directory / PROGRESS_LOG if mode in ("excel", "both") else None
        self.console = mode in ("console", "both")
        self.started = perf_counter()
        self.rows = []

    def record(self, stage, **details):
        if self.path is None and not self.console:
            return
        elapsed = round(perf_counter() - self.started, 3)
        event = {"elapsed_seconds": elapsed, "stage": stage, **details}
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, allow_nan=False) + "\n")
            self.rows.append(event)
        if self.console:
            message = f"[{elapsed:7.3f}s] {stage}"
            if details:
                message += (f" {details['model']} | candidate {details['candidate']}/{details['candidates']}"
                            f" | fold {details['fold']}/{details['folds']} | {details['metric']}={details['score']:.4f}")
            print(message, file=sys.stderr, flush=True)

    def fold(self, model, candidate, candidates, fold, folds, metric, score):
        self.record("cv", model=model, candidate=candidate, candidates=candidates,
                    fold=fold, folds=folds, metric=metric, score=float(score))

    def failed(self):
        with suppress(OSError):
            self.record("failed")

    def attach(self, sheets):
        if self.path is None:
            return ()
        sheets[PROGRESS_SHEET] = pd.DataFrame.from_records(self.rows, columns=FIELDS)
        return (PROGRESS_LOG,)


def validate_progress(directory, manifest, sheets):
    mode = progress_mode(manifest["settings"].get("progress", "off"))
    saved = mode in ("excel", "both") and manifest["status"] == "completed"
    if (PROGRESS_LOG in manifest["artifacts"]) != saved or (PROGRESS_SHEET in sheets) != saved:
        raise ValueError("Progress artifacts differ from the selected mode")
    if not saved:
        return set()

    from .ul_subsample_validation import same_table

    path = directory / PROGRESS_LOG
    if file_sha256(path) != manifest["artifacts"][PROGRESS_LOG]:
        raise ValueError("Progress log checksum differs")
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    expected = pd.DataFrame.from_records(rows, columns=FIELDS)
    if expected.empty or expected.stage.iloc[-1] != "report" or not expected.elapsed_seconds.is_monotonic_increasing:
        raise ValueError("Invalid progress event order")
    same_table(sheets[PROGRESS_SHEET].fillna(""), expected.fillna(""), "Progress sheet differs from the log")
    return {PROGRESS_SHEET}
