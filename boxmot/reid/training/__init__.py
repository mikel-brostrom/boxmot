"""ReID training utilities exposed without loading Torch at package import."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "BaseTrainer": ("boxmot.reid.training.base", "BaseTrainer"),
    "AugmentationConfig": ("boxmot.reid.training.config", "AugmentationConfig"),
    "DataConfig": ("boxmot.reid.training.config", "DataConfig"),
    "EvalConfig": ("boxmot.reid.training.config", "EvalConfig"),
    "LossConfig": ("boxmot.reid.training.config", "LossConfig"),
    "ModelConfig": ("boxmot.reid.training.config", "ModelConfig"),
    "OptimizationConfig": ("boxmot.reid.training.config", "OptimizationConfig"),
    "ReIDTrainConfig": ("boxmot.reid.training.config", "ReIDTrainConfig"),
    "RunConfig": ("boxmot.reid.training.config", "RunConfig"),
    "evaluate_ranking": ("boxmot.reid.training.evaluator", "evaluate_ranking"),
    "METRIC_LOSS_REGISTRY": ("boxmot.reid.training.losses", "METRIC_LOSS_REGISTRY"),
    "AdaSPLoss": ("boxmot.reid.training.losses", "AdaSPLoss"),
    "CenterLoss": ("boxmot.reid.training.losses", "CenterLoss"),
    "CrossEntropyLabelSmooth": ("boxmot.reid.training.losses", "CrossEntropyLabelSmooth"),
    "CrossScaleMajorityMarginLoss": (
        "boxmot.reid.training.losses",
        "CrossScaleMajorityMarginLoss",
    ),
    "MultiSimilarityLoss": ("boxmot.reid.training.losses", "MultiSimilarityLoss"),
    "TreeBoostAPLoss": ("boxmot.reid.training.losses", "TreeBoostAPLoss"),
    "TripletLoss": ("boxmot.reid.training.losses", "TripletLoss"),
    "WeightedRegularizedTripletLoss": (
        "boxmot.reid.training.losses",
        "WeightedRegularizedTripletLoss",
    ),
}

__all__ = tuple(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
