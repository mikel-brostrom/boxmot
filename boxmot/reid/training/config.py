"""Typed configuration objects for ReID training."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Tuple

TRAIN_HPARAM_SECTIONS = frozenset(
    {
        "run",
        "data",
        "model",
        "optimization",
        "losses",
        "augmentation",
        "evaluation",
        "system",
        "derived",
    }
)

TRAIN_HPARAM_TO_ARG = {
    "model_name": "model",
    "dataset": "dataset",
    "data_dir": "data_dir",
    "img_size": "imgsz",
    "preprocess": "preprocess",
    "loss_type": "loss",
    "pretrained": "pretrained",
    "epochs": "epochs",
    "batch_size": "batch_size",
    "lr": "lr",
    "weight_decay": "weight_decay",
    "eta_min": "eta_min",
    "warmup_epochs": "warmup_epochs",
    "label_smooth": "label_smooth",
    "classifier_loss": "classifier_loss",
    "margin": "margin",
    "triplet_soft_margin": "triplet_soft_margin",
    "arcface_scale": "arcface_scale",
    "arcface_margin": "arcface_margin",
    "cosface_scale": "cosface_scale",
    "cosface_margin": "cosface_margin",
    "center_loss_weight": "center_loss_weight",
    "id_loss_weight": "id_loss_weight",
    "metric_loss_weight": "metric_loss_weight",
    "aux_ce_weight": "aux_ce_weight",
    "aux_ce_drop_epoch": "aux_ce_drop_epoch",
    "branch_loss_agg": "branch_loss_agg",
    "metric_feature": "metric_feature",
    "inference_feature": "inference_feature",
    "feature_fusion": "feature_fusion",
    "feat_dim": "feat_dim",
    "neck_dim": "neck_dim",
    "drop_path_rate": "drop_path_rate",
    "attention_window_layout": "attention_window_layout",
    "attention_bias": "attention_bias",
    "attention_mask": "attention_mask",
    "attention_shift": "attention_shift",
    "stage3_global": "stage3_global",
    "reid_adapter_stages": "reid_adapter_stages",
    "reid_adapter_reduction": "reid_adapter_reduction",
    "head_pool": "head_pool",
    "head_parts": "head_parts",
    "head_type": "head_type",
    "part_pooling": "part_pooling",
    "num_part_tokens": "num_part_tokens",
    "decouple_patterns": "decouple_patterns",
    "pattern_adapter_dim": "pattern_adapter_dim",
    "stripe_visibility": "stripe_visibility",
    "branch_aware_metric": "branch_aware_metric",
    "branch_metric_part_weight": "branch_metric_part_weight",
    "head_warmup_epochs": "head_warmup_epochs",
    "head_warmup_lr_mult": "head_warmup_lr_mult",
    "vit_lr_profile": "vit_lr_profile",
    "backbone_freeze_epochs": "backbone_freeze_epochs",
    "p": "p_ids",
    "k": "k_instances",
    "seed": "seed",
    "deterministic": "deterministic",
    "device": "device",
    "num_workers": "num_workers",
    "ema_decay": "ema_decay",
    "gaussian_blur": "gaussian_blur",
    "random_grayscale": "random_grayscale",
    "color_jitter": "color_jitter",
    "random_erasing": "random_erasing",
    "random_patch": "random_patch",
    "color_augmentation": "color_augmentation",
    "flip_tta": "flip_tta",
    "eval_interval": "eval_interval",
}


def _nested_get(data: dict[str, Any], *keys: str) -> Any | None:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def flatten_train_hparams(hparams: dict[str, Any]) -> dict[str, Any]:
    """Normalize saved nested hparams into flat trainer hparam keys."""
    if not any(key in hparams for key in TRAIN_HPARAM_SECTIONS):
        return dict(hparams)

    flat = {key: value for key, value in hparams.items() if key not in TRAIN_HPARAM_SECTIONS}
    mappings = {
        "model_name": ("run", "model_name"),
        "seed": ("run", "seed"),
        "deterministic": ("run", "deterministic"),
        "pretrained": ("run", "pretrained"),
        "dataset": ("data", "dataset"),
        "data_dir": ("data", "data_dir"),
        "img_size": ("data", "img_size"),
        "preprocess": ("data", "preprocess"),
        "num_classes": ("data", "num_classes"),
        "batch_size": ("data", "batch_size"),
        "p": ("data", "sampler", "p"),
        "k": ("data", "sampler", "k"),
        "num_workers": ("data", "num_workers"),
        "is_vit": ("model", "is_vit"),
        "feature_fusion": ("model", "feature_fusion"),
        "feat_dim": ("model", "feat_dim"),
        "neck_dim": ("model", "neck_dim"),
        "attention_window_layout": ("model", "attention", "window_layout"),
        "attention_bias": ("model", "attention", "bias"),
        "attention_mask": ("model", "attention", "mask"),
        "attention_shift": ("model", "attention", "shift"),
        "stage3_global": ("model", "attention", "stage3_global"),
        "reid_adapter_stages": ("model", "reid_adapters", "stages"),
        "reid_adapter_reduction": ("model", "reid_adapters", "reduction"),
        "head_pool": ("model", "head", "pool"),
        "head_parts": ("model", "head", "parts"),
        "head_type": ("model", "head", "head_type"),
        "part_pooling": ("model", "head", "part_pooling"),
        "num_part_tokens": ("model", "head", "num_part_tokens"),
        "decouple_patterns": ("model", "head", "decouple_patterns"),
        "pattern_adapter_dim": ("model", "head", "pattern_adapter_dim"),
        "stripe_visibility": ("model", "head", "stripe_visibility"),
        "head_warmup_epochs": ("model", "head", "warmup_epochs"),
        "head_warmup_lr_mult": ("model", "head", "warmup_lr_mult"),
        "metric_feature": ("model", "feature_selection", "metric_feature"),
        "inference_feature": ("model", "feature_selection", "inference_feature"),
        "branch_aware_metric": ("model", "branch", "aware_metric"),
        "branch_metric_part_weight": ("model", "branch", "metric_part_weight"),
        "branch_loss_agg": ("model", "branch", "loss_agg"),
        "drop_path_rate": ("model", "regularization", "drop_path_rate"),
        "epochs": ("optimization", "epochs"),
        "optimizer": ("optimization", "optimizer"),
        "lr": ("optimization", "lr"),
        "weight_decay": ("optimization", "weight_decay"),
        "grad_clip": ("optimization", "grad_clip"),
        "layer_decay": ("optimization", "layer_decay"),
        "vit_lr_profile": ("optimization", "vit_lr_profile"),
        "backbone_freeze_epochs": ("optimization", "backbone_freeze_epochs"),
        "scheduler": ("optimization", "scheduler", "name"),
        "eta_min": ("optimization", "scheduler", "eta_min"),
        "warmup_epochs": ("optimization", "scheduler", "warmup_epochs"),
        "ema_decay": ("optimization", "ema_decay"),
        "loss_type": ("losses", "loss_type"),
        "classifier_loss": ("losses", "classifier_loss"),
        "label_smooth": ("losses", "label_smooth"),
        "margin": ("losses", "triplet", "margin"),
        "triplet_soft_margin": ("losses", "triplet", "soft_margin"),
        "soft_margin_triplet": ("losses", "triplet", "soft_margin"),
        "id_loss_weight": ("losses", "weights", "id_loss_weight"),
        "metric_loss_weight": ("losses", "weights", "metric_loss_weight"),
        "center_loss_weight": ("losses", "weights", "center_loss_weight"),
        "aux_ce_weight": ("losses", "weights", "aux_ce_weight"),
        "aux_ce_drop_epoch": ("losses", "aux_ce_drop_epoch"),
        "arcface_scale": ("losses", "arcface", "scale"),
        "arcface_margin": ("losses", "arcface", "margin"),
        "cosface_scale": ("losses", "cosface", "scale"),
        "cosface_margin": ("losses", "cosface", "margin"),
        "color_jitter": ("augmentation", "color_jitter"),
        "gaussian_blur": ("augmentation", "gaussian_blur"),
        "random_grayscale": ("augmentation", "random_grayscale"),
        "random_erasing": ("augmentation", "random_erasing"),
        "random_patch": ("augmentation", "random_patch"),
        "color_augmentation": ("augmentation", "color_augmentation"),
        "eval_interval": ("evaluation", "eval_interval"),
        "eval_datasets": ("evaluation", "eval_datasets"),
        "flip_tta": ("evaluation", "flip_tta"),
        "device": ("system", "device"),
        "metric_dim": ("derived", "metric_dim"),
        "classifier_dim": ("derived", "classifier_dim"),
        "n_params": ("derived", "n_params"),
    }

    for key, path in mappings.items():
        value = _nested_get(hparams, *path)
        if value is not None:
            flat[key] = value

    return flat


def load_train_hparams(resume_path: str | Path) -> dict[str, Any]:
    """Load normalized train hparams from a resume directory or checkpoint path."""
    path = Path(resume_path)
    hparams_file = path / "hparams.json" if path.is_dir() else path.parent / "hparams.json"
    if not hparams_file.exists():
        return {}
    return flatten_train_hparams(json.loads(hparams_file.read_text(encoding="utf-8")))


def _arg_or_hparam(
    args: Any,
    hparams: dict[str, Any],
    explicit_keys: set[str],
    hparam_key: str,
    arg_key: str,
    default: Any = None,
) -> Any:
    if hparam_key in hparams and arg_key not in explicit_keys:
        return hparams[hparam_key]
    return getattr(args, arg_key, default)


def _normalize_img_size(img_size: Any) -> tuple[int, int]:
    if isinstance(img_size, int):
        return (img_size, img_size // 2)
    if isinstance(img_size, (list, tuple)) and len(img_size) == 1:
        return (img_size[0], img_size[0] // 2)
    return tuple(img_size)


def trainer_kwargs_from_args(
    args: Any,
    hparams: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build trainer kwargs from a CLI/API namespace and optional resume hparams."""
    explicit_keys = set(getattr(args, "train_explicit_keys", ()))
    resume = getattr(args, "resume", None)
    if hparams is None:
        hparams = load_train_hparams(resume) if resume else {}

    def value(hparam_key: str, arg_key: str | None = None, default: Any = None) -> Any:
        resolved_arg_key = arg_key or TRAIN_HPARAM_TO_ARG.get(hparam_key, hparam_key)
        return _arg_or_hparam(args, hparams, explicit_keys, hparam_key, resolved_arg_key, default)

    img_size = _normalize_img_size(value("img_size", "imgsz", (256, 128)))
    data_dir = value("data_dir", "data_dir")
    if not data_dir:
        raise ValueError("--data-dir is required (not found in hparams.json either)")

    return {
        "model_name": value("model_name", "model"),
        "dataset_name": value("dataset", "dataset"),
        "data_dir": data_dir,
        "loss_type": value("loss_type", "loss", "triplet"),
        "preprocess": value("preprocess", "preprocess", "resize"),
        "img_size": img_size,
        "batch_size": value("batch_size", "batch_size", 64),
        "lr": value("lr", "lr", 3.5e-4),
        "weight_decay": value("weight_decay", "weight_decay", 5e-4),
        "epochs": value("epochs", "epochs", 120),
        "warmup_epochs": value("warmup_epochs", "warmup_epochs", 10),
        "eval_interval": value("eval_interval", "eval_interval", 10),
        "p": value("p", "p_ids", 16),
        "k": value("k", "k_instances", 4),
        "margin": value("margin", "margin", 0.3),
        "label_smooth": value("label_smooth", "label_smooth", 0.1),
        "classifier_loss": value("classifier_loss", "classifier_loss", "ce"),
        "triplet_soft_margin": value("triplet_soft_margin", "triplet_soft_margin"),
        "arcface_scale": value("arcface_scale", "arcface_scale", 30.0),
        "arcface_margin": value("arcface_margin", "arcface_margin", 0.5),
        "cosface_scale": value("cosface_scale", "cosface_scale", 30.0),
        "cosface_margin": value("cosface_margin", "cosface_margin", 0.35),
        "center_loss_weight": value("center_loss_weight", "center_loss_weight", 5e-4),
        "id_loss_weight": value("id_loss_weight", "id_loss_weight", 1.0),
        "metric_loss_weight": value("metric_loss_weight", "metric_loss_weight", 1.0),
        "aux_ce_weight": value("aux_ce_weight", "aux_ce_weight", 1.0),
        "aux_ce_drop_epoch": value("aux_ce_drop_epoch", "aux_ce_drop_epoch", 0),
        "branch_loss_agg": value("branch_loss_agg", "branch_loss_agg", "mean"),
        "metric_feature": value("metric_feature", "metric_feature", "auto"),
        "inference_feature": value("inference_feature", "inference_feature", "concat_bn"),
        "feature_fusion": value("feature_fusion", "feature_fusion", "last3"),
        "feat_dim": value("feat_dim", "feat_dim", 512),
        "neck_dim": value("neck_dim", "neck_dim", 512),
        "drop_path_rate": value("drop_path_rate", "drop_path_rate", 0.1),
        "attention_window_layout": value("attention_window_layout", "attention_window_layout", "legacy"),
        "attention_bias": value("attention_bias", "attention_bias", "absolute"),
        "attention_mask": value("attention_mask", "attention_mask", False),
        "attention_shift": value("attention_shift", "attention_shift", False),
        "stage3_global": value("stage3_global", "stage3_global", False),
        "reid_adapter_stages": value("reid_adapter_stages", "reid_adapter_stages", ()),
        "reid_adapter_reduction": value("reid_adapter_reduction", "reid_adapter_reduction", 4),
        "head_pool": value("head_pool", "head_pool", "avg"),
        "head_parts": value("head_parts", "head_parts", (1, 2)),
        "head_type": value("head_type", "head_type", "standard"),
        "part_pooling": value("part_pooling", "part_pooling", "stripes"),
        "num_part_tokens": value("num_part_tokens", "num_part_tokens", 4),
        "decouple_patterns": value("decouple_patterns", "decouple_patterns", False),
        "pattern_adapter_dim": value("pattern_adapter_dim", "pattern_adapter_dim", 128),
        "stripe_visibility": value("stripe_visibility", "stripe_visibility", False),
        "branch_aware_metric": value("branch_aware_metric", "branch_aware_metric", False),
        "branch_metric_part_weight": value("branch_metric_part_weight", "branch_metric_part_weight", 0.5),
        "head_warmup_epochs": value("head_warmup_epochs", "head_warmup_epochs", 0),
        "head_warmup_lr_mult": value("head_warmup_lr_mult", "head_warmup_lr_mult", 2.0),
        "vit_lr_profile": value("vit_lr_profile", "vit_lr_profile", "layer_decay"),
        "backbone_freeze_epochs": value("backbone_freeze_epochs", "backbone_freeze_epochs", 0),
        "eta_min": value("eta_min", "eta_min", 1e-7),
        "pretrained": value("pretrained", "pretrained", True),
        "device": value("device", "device", "cpu"),
        "project": str(value("project", "project", "runs/reid_train")),
        "name": value("name", "name", "exp"),
        "num_workers": value("num_workers", "num_workers", 4),
        "seed": value("seed", "seed", 0),
        "deterministic": value("deterministic", "deterministic", True),
        "eval_datasets": value("eval_datasets", "eval_datasets"),
        "ema_decay": value("ema_decay", "ema_decay"),
        "gaussian_blur": value("gaussian_blur", "gaussian_blur", False),
        "random_grayscale": value("random_grayscale", "random_grayscale", 0.0),
        "color_jitter": value("color_jitter", "color_jitter", False),
        "random_erasing": value("random_erasing", "random_erasing", 0.5),
        "random_patch": value("random_patch", "random_patch", True),
        "color_augmentation": value("color_augmentation", "color_augmentation", True),
        "flip_tta": value("flip_tta", "flip_tta"),
        "resume": resume,
        "explicit_hparams": explicit_keys,
    }


@dataclass(frozen=True)
class RunConfig:
    """Execution, persistence, and reproducibility settings."""

    device: str = "cpu"
    project: str = "runs/reid_train"
    name: str = "exp"
    seed: int = 0
    deterministic: bool = True
    resume: Optional[str] = None
    explicit_hparams: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class DataConfig:
    """Dataset, preprocessing, sampling, and loader settings."""

    dataset_name: str = "market1501"
    data_dir: str = "."
    preprocess: str = "resize"
    img_size: Tuple[int, int] = (256, 128)
    batch_size: int = 64
    p: int = 16
    k: int = 4
    num_workers: int = 4


@dataclass(frozen=True)
class ModelConfig:
    """Backbone and embedding-head settings."""

    model_name: str = "osnet_x0_25"
    pretrained: bool = True
    metric_feature: str = "auto"
    inference_feature: str = "concat_bn"
    feature_fusion: str = "last3"
    feat_dim: int = 512
    neck_dim: int = 512
    drop_path_rate: float = 0.1
    attention_window_layout: str = "legacy"
    attention_bias: str = "absolute"
    attention_mask: bool = False
    attention_shift: bool = False
    stage3_global: bool = False
    reid_adapter_stages: tuple[int, ...] | list[int] = ()
    reid_adapter_reduction: int = 4
    branch_aware_metric: bool = False
    branch_metric_part_weight: float = 0.5
    head_pool: str = "avg"
    head_parts: tuple[int, ...] | list[int] = (1, 2)
    head_type: str = "standard"
    part_pooling: str = "stripes"
    num_part_tokens: int = 4
    decouple_patterns: bool = False
    pattern_adapter_dim: int = 128
    stripe_visibility: bool = False
    head_warmup_epochs: int = 0
    head_warmup_lr_mult: float = 2.0


@dataclass(frozen=True)
class LossConfig:
    """Classification, metric, center, and branch-loss settings."""

    loss_type: str = "triplet"
    margin: float = 0.3
    label_smooth: float = 0.1
    classifier_loss: str = "ce"
    triplet_soft_margin: Optional[bool] = None
    arcface_scale: float = 30.0
    arcface_margin: float = 0.5
    cosface_scale: float = 30.0
    cosface_margin: float = 0.35
    center_loss_weight: float = 5e-4
    id_loss_weight: float = 1.0
    metric_loss_weight: float = 1.0
    aux_ce_weight: float = 1.0
    aux_ce_drop_epoch: int = 0
    branch_loss_agg: str = "mean"


@dataclass(frozen=True)
class OptimizationConfig:
    """Optimizer and scheduler settings."""

    lr: float = 3.5e-4
    weight_decay: float = 5e-4
    epochs: int = 120
    warmup_epochs: int = 10
    eta_min: float = 1e-7
    ema_decay: Optional[float] = None
    vit_lr_profile: str = "layer_decay"
    backbone_freeze_epochs: int = 0


@dataclass(frozen=True)
class AugmentationConfig:
    """Training-time image augmentation settings."""

    gaussian_blur: bool = False
    random_grayscale: float = 0.0
    color_jitter: bool = False
    random_erasing: float = 0.5
    random_patch: bool = True
    color_augmentation: bool = True


@dataclass(frozen=True)
class EvalConfig:
    """Validation frequency and inference augmentation settings."""

    eval_interval: int = 10
    eval_datasets: tuple[str, ...] = ()
    flip_tta: Optional[bool] = None


@dataclass(frozen=True)
class ReIDTrainConfig:
    """Complete typed ReID training configuration."""

    run: RunConfig = field(default_factory=RunConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)

    @classmethod
    def from_flat_kwargs(cls, **values) -> "ReIDTrainConfig":
        """Build nested configuration from the legacy trainer keyword surface."""
        explicit = values.get("explicit_hparams") or ()
        image_size = values.get("img_size", (256, 128))
        if isinstance(image_size, int):
            image_size = (image_size, image_size // 2)
        return cls(
            run=RunConfig(
                device=values.get("device", "cpu"),
                project=str(values.get("project", "runs/reid_train")),
                name=values.get("name", "exp"),
                seed=values.get("seed", 0),
                deterministic=values.get("deterministic", True),
                resume=values.get("resume"),
                explicit_hparams=set(explicit),
            ),
            data=DataConfig(
                dataset_name=values["dataset_name"],
                data_dir=str(values["data_dir"]),
                preprocess=values.get("preprocess", "resize"),
                img_size=tuple(image_size),
                batch_size=values.get("batch_size", 64),
                p=values.get("p", 16),
                k=values.get("k", 4),
                num_workers=values.get("num_workers", 4),
            ),
            model=ModelConfig(
                model_name=values["model_name"],
                pretrained=values.get("pretrained", True),
                metric_feature=values.get("metric_feature", "auto"),
                inference_feature=values.get("inference_feature", "concat_bn"),
                feature_fusion=values.get("feature_fusion", "last3"),
                feat_dim=values.get("feat_dim", 512),
                neck_dim=values.get("neck_dim", 512),
                drop_path_rate=values.get("drop_path_rate", 0.1),
                attention_window_layout=values.get("attention_window_layout", "legacy"),
                attention_bias=values.get("attention_bias", "absolute"),
                attention_mask=values.get("attention_mask", False),
                attention_shift=values.get("attention_shift", False),
                stage3_global=values.get("stage3_global", False),
                reid_adapter_stages=values.get("reid_adapter_stages", ()),
                reid_adapter_reduction=values.get("reid_adapter_reduction", 4),
                branch_aware_metric=values.get("branch_aware_metric", False),
                branch_metric_part_weight=values.get("branch_metric_part_weight", 0.5),
                head_pool=values.get("head_pool", "avg"),
                head_parts=values.get("head_parts", (1, 2)),
                head_type=values.get("head_type", "standard"),
                part_pooling=values.get("part_pooling", "stripes"),
                num_part_tokens=values.get("num_part_tokens", 4),
                decouple_patterns=values.get("decouple_patterns", False),
                pattern_adapter_dim=values.get("pattern_adapter_dim", 128),
                stripe_visibility=values.get("stripe_visibility", False),
                head_warmup_epochs=values.get("head_warmup_epochs", 0),
                head_warmup_lr_mult=values.get("head_warmup_lr_mult", 2.0),
            ),
            loss=LossConfig(
                loss_type=values.get("loss_type", "triplet"),
                margin=values.get("margin", 0.3),
                label_smooth=values.get("label_smooth", 0.1),
                classifier_loss=values.get("classifier_loss", "ce"),
                triplet_soft_margin=values.get("triplet_soft_margin"),
                arcface_scale=values.get("arcface_scale", 30.0),
                arcface_margin=values.get("arcface_margin", 0.5),
                cosface_scale=values.get("cosface_scale", 30.0),
                cosface_margin=values.get("cosface_margin", 0.35),
                center_loss_weight=values.get("center_loss_weight", 5e-4),
                id_loss_weight=values.get("id_loss_weight", 1.0),
                metric_loss_weight=values.get("metric_loss_weight", 1.0),
                aux_ce_weight=values.get("aux_ce_weight", 1.0),
                aux_ce_drop_epoch=values.get("aux_ce_drop_epoch", 0),
                branch_loss_agg=values.get("branch_loss_agg", "mean"),
            ),
            optimization=OptimizationConfig(
                lr=values.get("lr", 3.5e-4),
                weight_decay=values.get("weight_decay", 5e-4),
                epochs=values.get("epochs", 120),
                warmup_epochs=values.get("warmup_epochs", 10),
                eta_min=values.get("eta_min", 1e-7),
                ema_decay=values.get("ema_decay"),
                vit_lr_profile=values.get("vit_lr_profile", "layer_decay"),
                backbone_freeze_epochs=values.get("backbone_freeze_epochs", 0),
            ),
            augmentation=AugmentationConfig(
                gaussian_blur=values.get("gaussian_blur", False),
                random_grayscale=values.get("random_grayscale", 0.0),
                color_jitter=values.get("color_jitter", False),
                random_erasing=values.get("random_erasing", 0.5),
                random_patch=values.get("random_patch", True),
                color_augmentation=values.get("color_augmentation", True),
            ),
            evaluation=EvalConfig(
                eval_interval=values.get("eval_interval", 10),
                eval_datasets=tuple(values.get("eval_datasets") or ()),
                flip_tta=values.get("flip_tta"),
            ),
        )

    def to_trainer_kwargs(self) -> dict:
        """Flatten nested configuration for the compatibility constructor."""
        return {
            "model_name": self.model.model_name,
            "dataset_name": self.data.dataset_name,
            "data_dir": self.data.data_dir,
            **self.loss.__dict__,
            "preprocess": self.data.preprocess,
            "img_size": self.data.img_size,
            "batch_size": self.data.batch_size,
            "p": self.data.p,
            "k": self.data.k,
            "num_workers": self.data.num_workers,
            **self.optimization.__dict__,
            **self.augmentation.__dict__,
            "eval_interval": self.evaluation.eval_interval,
            "eval_datasets": list(self.evaluation.eval_datasets),
            "flip_tta": self.evaluation.flip_tta,
            **self.model.__dict__,
            **self.run.__dict__,
        }
