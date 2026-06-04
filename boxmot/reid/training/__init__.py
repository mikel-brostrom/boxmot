"""ReID model training utilities: losses, trainer, and evaluation."""

from boxmot.reid.training.evaluator import evaluate_ranking
from boxmot.reid.training.losses import (
    CenterLoss,
    CrossEntropyLabelSmooth,
    METRIC_LOSS_REGISTRY,
    MultiSimilarityLoss,
    TripletLoss,
)

__all__ = (
    "CenterLoss",
    "CrossEntropyLabelSmooth",
    "METRIC_LOSS_REGISTRY",
    "MultiSimilarityLoss",
    "TripletLoss",
    "evaluate_ranking",
)
