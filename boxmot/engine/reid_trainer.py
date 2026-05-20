"""Engine entry point for ReID model training.

Invoked by the CLI ``train`` subcommand via ``main(args)``.
"""

from __future__ import annotations

from boxmot.reid.training.trainer import ReIDTrainer, TrainResult
from boxmot.utils import logger as LOGGER


def main(args) -> TrainResult:
    """Train a ReID model.

    Args:
        args: Namespace with training parameters (from CLI or Python API).

    Returns:
        TrainResult with best epoch, metrics, and saved weights path.
    """
    img_size = getattr(args, "imgsz", (256, 128))
    if isinstance(img_size, int):
        img_size = (img_size, img_size // 2)
    elif isinstance(img_size, (list, tuple)) and len(img_size) == 1:
        img_size = (img_size[0], img_size[0] // 2)

    trainer = ReIDTrainer(
        model_name=args.model,
        dataset_name=args.dataset,
        data_dir=args.data_dir,
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
        resume=getattr(args, "resume", None),
    )

    result = trainer.run()
    LOGGER.info(
        f"Training finished. Best weights: {result.weights_path}  "
        f"mAP={result.best_mAP:.2%}  R1={result.best_rank1:.2%}"
    )
    return result
