"""Typed configuration objects for ReID training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


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
