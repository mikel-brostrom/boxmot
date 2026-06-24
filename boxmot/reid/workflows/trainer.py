"""Workflow entry point for ReID model training.

Invoked by the CLI ``train`` subcommand via ``main(args)``.
"""

from __future__ import annotations

import json
from pathlib import Path

from boxmot.reid.training.config import ReIDTrainConfig
from boxmot.reid.training.trainer import ReIDTrainer, TrainResult
from boxmot.utils import logger as LOGGER


def _nested_get(data: dict, *keys):
    cur = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _flatten_hparams_for_resume(hparams: dict) -> dict:
    """Normalize nested hparams.json layout into legacy flat keys."""
    sections = {
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
    if not any(key in hparams for key in sections):
        return hparams

    flat: dict = {}

    # Keep any non-section top-level keys as-is.
    for key, value in hparams.items():
        if key not in sections:
            flat[key] = value

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


def _load_hparams_from_resume(resume_path: str | Path) -> dict:
    """Load hparams.json from a resume directory or checkpoint file path."""
    p = Path(resume_path)
    hparams_file = p / "hparams.json" if p.is_dir() else p.parent / "hparams.json"
    if hparams_file.exists():
        with open(hparams_file) as f:
            return _flatten_hparams_for_resume(json.load(f))
    return {}


def main(args) -> TrainResult:
    """Train a ReID model.

    Args:
        args: Namespace with training parameters (from CLI or Python API).

    Returns:
        TrainResult with best epoch, metrics, and saved weights path.
    """
    explicit_keys = set(getattr(args, "train_explicit_keys", ()))

    # When resuming, load saved hparams and use them as defaults for any
    # parameters the user didn't explicitly override on the CLI.
    resume = getattr(args, "resume", None)
    if resume:
        hparams = _load_hparams_from_resume(resume)
        if hparams:
            LOGGER.info(
                f"Loaded hparams from resume dir: "
                f"model={hparams.get('model_name')}, dataset={hparams.get('dataset')}"
            )
            # Map hparams.json keys → args attribute names
            _HPARAM_TO_ARG = {
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
            for hp_key, arg_key in _HPARAM_TO_ARG.items():
                if hp_key in hparams and arg_key not in explicit_keys:
                    setattr(args, arg_key, hparams[hp_key])

    img_size = getattr(args, "imgsz", (256, 128))
    if isinstance(img_size, int):
        img_size = (img_size, img_size // 2)
    elif isinstance(img_size, (list, tuple)) and len(img_size) == 1:
        img_size = (img_size[0], img_size[0] // 2)

    data_dir = getattr(args, "data_dir", None)
    if not data_dir:
        raise ValueError("--data-dir is required (not found in hparams.json either)")

    trainer_kwargs = dict(
        model_name=args.model,
        dataset_name=args.dataset,
        data_dir=data_dir,
        loss_type=getattr(args, "loss", "triplet"),
        preprocess=getattr(args, "preprocess", "resize"),
        img_size=tuple(img_size),
        batch_size=getattr(args, "batch_size", 64),
        lr=getattr(args, "lr", 3.5e-4),
        weight_decay=getattr(args, "weight_decay", 5e-4),
        epochs=getattr(args, "epochs", 120),
        warmup_epochs=getattr(args, "warmup_epochs", 10),
        eval_interval=getattr(args, "eval_interval", 10),
        p=getattr(args, "p_ids", 16),
        k=getattr(args, "k_instances", 4),
        margin=getattr(args, "margin", 0.3),
        label_smooth=getattr(args, "label_smooth", 0.1),
        classifier_loss=getattr(args, "classifier_loss", "ce"),
        triplet_soft_margin=getattr(args, "triplet_soft_margin", None),
        arcface_scale=getattr(args, "arcface_scale", 30.0),
        arcface_margin=getattr(args, "arcface_margin", 0.5),
        cosface_scale=getattr(args, "cosface_scale", 30.0),
        cosface_margin=getattr(args, "cosface_margin", 0.35),
        center_loss_weight=getattr(args, "center_loss_weight", 5e-4),
        id_loss_weight=getattr(args, "id_loss_weight", 1.0),
        metric_loss_weight=getattr(args, "metric_loss_weight", 1.0),
        aux_ce_weight=getattr(args, "aux_ce_weight", 1.0),
        aux_ce_drop_epoch=getattr(args, "aux_ce_drop_epoch", 0),
        branch_loss_agg=getattr(args, "branch_loss_agg", "mean"),
        metric_feature=getattr(args, "metric_feature", "auto"),
        inference_feature=getattr(args, "inference_feature", "concat_bn"),
        feature_fusion=getattr(args, "feature_fusion", "last3"),
        feat_dim=getattr(args, "feat_dim", 512),
        neck_dim=getattr(args, "neck_dim", 512),
        drop_path_rate=getattr(args, "drop_path_rate", 0.1),
        attention_window_layout=getattr(args, "attention_window_layout", "legacy"),
        attention_bias=getattr(args, "attention_bias", "absolute"),
        attention_mask=getattr(args, "attention_mask", False),
        attention_shift=getattr(args, "attention_shift", False),
        stage3_global=getattr(args, "stage3_global", False),
        reid_adapter_stages=getattr(args, "reid_adapter_stages", ()),
        reid_adapter_reduction=getattr(args, "reid_adapter_reduction", 4),
        head_pool=getattr(args, "head_pool", "avg"),
        head_parts=getattr(args, "head_parts", (1, 2)),
        head_type=getattr(args, "head_type", "standard"),
        part_pooling=getattr(args, "part_pooling", "stripes"),
        num_part_tokens=getattr(args, "num_part_tokens", 4),
        decouple_patterns=getattr(args, "decouple_patterns", False),
        pattern_adapter_dim=getattr(args, "pattern_adapter_dim", 128),
        stripe_visibility=getattr(args, "stripe_visibility", False),
        branch_aware_metric=getattr(args, "branch_aware_metric", False),
        branch_metric_part_weight=getattr(args, "branch_metric_part_weight", 0.5),
        head_warmup_epochs=getattr(args, "head_warmup_epochs", 0),
        head_warmup_lr_mult=getattr(args, "head_warmup_lr_mult", 2.0),
        vit_lr_profile=getattr(args, "vit_lr_profile", "layer_decay"),
        backbone_freeze_epochs=getattr(args, "backbone_freeze_epochs", 0),
        eta_min=getattr(args, "eta_min", 1e-7),
        pretrained=getattr(args, "pretrained", True),
        device=getattr(args, "device", "cpu"),
        project=str(getattr(args, "project", "runs/reid_train")),
        name=getattr(args, "name", "exp"),
        num_workers=getattr(args, "num_workers", 4),
        seed=getattr(args, "seed", 0),
        deterministic=getattr(args, "deterministic", True),
        eval_datasets=getattr(args, "eval_datasets", None),
        ema_decay=getattr(args, "ema_decay", None),
        gaussian_blur=getattr(args, "gaussian_blur", False),
        random_grayscale=getattr(args, "random_grayscale", 0.0),
        color_jitter=getattr(args, "color_jitter", False),
        random_erasing=getattr(args, "random_erasing", 0.5),
        random_patch=getattr(args, "random_patch", True),
        color_augmentation=getattr(args, "color_augmentation", True),
        flip_tta=getattr(args, "flip_tta", None),
        resume=getattr(args, "resume", None),
        explicit_hparams=explicit_keys,
    )
    config = ReIDTrainConfig.from_flat_kwargs(**trainer_kwargs)
    trainer = (
        ReIDTrainer.from_config(config)
        if hasattr(ReIDTrainer, "from_config")
        else ReIDTrainer(**config.to_trainer_kwargs())
    )

    result = trainer.run()
    LOGGER.info(
        f"Training finished. Best weights: {result.weights_path}  "
        f"mAP={result.best_mAP:.2%}  R1={result.best_rank1:.2%}"
    )
    return result
