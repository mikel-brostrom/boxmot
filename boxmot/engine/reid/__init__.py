"""Engine-owned ReID command entrypoints for training, evaluation, comparison, and export."""

from boxmot.engine.reid.base import BasePredictor, BaseValidator

__all__ = (
    "BasePredictor",
    "BaseValidator",
    "comparison",
    "evaluator",
    "export",
    "trainer",
)
