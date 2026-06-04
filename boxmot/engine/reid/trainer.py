"""Engine entry point for ReID model training.

Invoked by the CLI ``train`` subcommand via ``main(args)``.
"""

from __future__ import annotations

import json
from pathlib import Path

from boxmot.reid.training.trainer import ReIDTrainer, TrainResult
from boxmot.utils import logger as LOGGER


def _load_hparams_from_resume(resume_path: str | Path) -> dict:
    """Load hparams.json from a resume directory or checkpoint file path."""
    p = Path(resume_path)
    hparams_file = p / "hparams.json" if p.is_dir() else p.parent / "hparams.json"
    if hparams_file.exists():
        with open(hparams_file) as f:
            return json.load(f)
    return {}


def main(args) -> TrainResult:
    """Train a ReID model.

    Args:
        args: Namespace with training parameters (from CLI or Python API).

    Returns:
        TrainResult with best epoch, metrics, and saved weights path.
    """
    # When resuming, load saved hparams and use them as defaults for any
    # parameters the user didn't explicitly override on the CLI.
    resume = getattr(args, "resume", None)
    if resume:
        hparams = _load_hparams_from_resume(resume)
        if hparams:
            LOGGER.info(f"Loaded hparams from resume dir: model={hparams.get('model_name')}, dataset={hparams.get('dataset')}")
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
                "margin": "margin",
                "center_loss_weight": "center_loss_weight",
                "p": "p_ids",
                "k": "k_instances",
                "seed": "seed",
                "device": "device",
                "num_workers": "num_workers",
                "ema_decay": "ema_decay",
                "gaussian_blur": "gaussian_blur",
                "random_grayscale": "random_grayscale",
                "color_jitter": "color_jitter",
                "random_erasing": "random_erasing",
                "eval_interval": "eval_interval",
            }
            for hp_key, arg_key in _HPARAM_TO_ARG.items():
                if hp_key in hparams:
                    setattr(args, arg_key, hparams[hp_key])

    img_size = getattr(args, "imgsz", (256, 128))
    if isinstance(img_size, int):
        img_size = (img_size, img_size // 2)
    elif isinstance(img_size, (list, tuple)) and len(img_size) == 1:
        img_size = (img_size[0], img_size[0] // 2)

    data_dir = getattr(args, "data_dir", None)
    if not data_dir:
        raise ValueError("--data-dir is required (not found in hparams.json either)")

    trainer = ReIDTrainer(
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
        center_loss_weight=getattr(args, "center_loss_weight", 5e-4),
        eta_min=getattr(args, "eta_min", 1e-7),
        pretrained=getattr(args, "pretrained", True),
        device=getattr(args, "device", "cpu"),
        project=str(getattr(args, "project", "runs/reid_train")),
        name=getattr(args, "name", "exp"),
        num_workers=getattr(args, "num_workers", 4),
        seed=getattr(args, "seed", 42),
        eval_datasets=getattr(args, "eval_datasets", None),
        ema_decay=getattr(args, "ema_decay", None),
        gaussian_blur=getattr(args, "gaussian_blur", False),
        random_grayscale=getattr(args, "random_grayscale", 0.0),
        color_jitter=getattr(args, "color_jitter", False),
        random_erasing=getattr(args, "random_erasing", 0.5),
        resume=getattr(args, "resume", None),
    )

    result = trainer.run()
    LOGGER.info(
        f"Training finished. Best weights: {result.weights_path}  "
        f"mAP={result.best_mAP:.2%}  R1={result.best_rank1:.2%}"
    )
    return result
