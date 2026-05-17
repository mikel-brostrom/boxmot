# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

"""Public BoxMOT Python API."""

from boxmot.engine.research import ResearchResult
from boxmot.engine.workflow_results import (
    ExportResult,
    GenerateResult,
    TrackRunResult,
    TuneResult,
    TuneTrialResult,
    ValidationResult,
)
from boxmot.reid.training.trainer import TrainResult

from ._facade import Boxmot, evaluate, track

__all__ = (
    "Boxmot",
    "ExportResult",
    "GenerateResult",
    "ResearchResult",
    "TrackRunResult",
    "TrainResult",
    "TuneResult",
    "TuneTrialResult",
    "ValidationResult",
    "evaluate",
    "track",
)
