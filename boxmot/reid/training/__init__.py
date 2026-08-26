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
    AdaSPLoss,
    CenterLoss,
    CrossEntropyLabelSmooth,
    CrossScaleMajorityMarginLoss,
    MultiSimilarityLoss,
    TreeBoostAPLoss,
    TripletLoss,
    WeightedRegularizedTripletLoss,
)

__all__ = (
    "AdaSPLoss",
    "CenterLoss",
    "CrossEntropyLabelSmooth",
    "CrossScaleMajorityMarginLoss",
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
    "TreeBoostAPLoss",
    "TripletLoss",
    "WeightedRegularizedTripletLoss",
    "evaluate_ranking",
)
