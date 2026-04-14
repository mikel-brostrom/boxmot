# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

"""Public BoxMOT Python API."""

from boxmot.engine.results import Results, Tracks

from ._facade import Boxmot, evaluate, track
from ._reporting import (
    CLI_RESULTS_SUMMARY_TITLE,
    CLI_TUNE_BEST_SUMMARY_TITLE,
    DEFAULT_TUNE_BEST_REPORT_TITLE,
    DEFAULT_VALIDATION_REPORT_TITLE,
)
from ._results import ExportResult, TrackRunResult, TuneResult, TuneTrialResult, ValidationResult

__all__ = (
    "Boxmot",
    "CLI_RESULTS_SUMMARY_TITLE",
    "CLI_TUNE_BEST_SUMMARY_TITLE",
    "DEFAULT_TUNE_BEST_REPORT_TITLE",
    "DEFAULT_VALIDATION_REPORT_TITLE",
    "ExportResult",
    "Results",
    "TrackRunResult",
    "Tracks",
    "TuneResult",
    "TuneTrialResult",
    "ValidationResult",
    "evaluate",
    "track",
)

