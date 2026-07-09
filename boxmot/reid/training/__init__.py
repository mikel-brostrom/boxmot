"""ReID model training utilities: losses, trainer, and evaluation."""

from boxmot.reid.training.base import BaseTrainer
from boxmot.reid.training.config import (
    AugmentationConfig,
    DataConfig,
    EvalConfig,
    LossConfig,
    ModelConfig,
    OptimizationConfig,
    ReIDTrainConfig,
    RunConfig,
)
from boxmot.reid.training.evaluator import evaluate_ranking
from boxmot.reid.training.losses import (
    METRIC_LOSS_REGISTRY,
    CenterLoss,
    CrossEntropyLabelSmooth,
    MultiSimilarityLoss,
    TripletLoss,
)

__all__ = (
    "CenterLoss",
    "CrossEntropyLabelSmooth",
    "AugmentationConfig",
    "BaseTrainer",
    "DataConfig",
    "EvalConfig",
    "LossConfig",
    "METRIC_LOSS_REGISTRY",
    "ModelConfig",
    "MultiSimilarityLoss",
    "OptimizationConfig",
    "ReIDTrainConfig",
    "RunConfig",
    "TripletLoss",
    "evaluate_ranking",
)
