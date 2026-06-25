"""Engine entry point for ReID model training.

Invoked by the CLI ``train`` subcommand via ``main(args)``.
"""

from __future__ import annotations

from boxmot.reid.training.config import (
    ReIDTrainConfig,
    load_train_hparams,
    trainer_kwargs_from_args,
)
from boxmot.reid.training.trainer import ReIDTrainer, TrainResult
from boxmot.utils import logger as LOGGER


def main(args) -> TrainResult:
    """Train a ReID model.

    Args:
        args: Namespace with training parameters (from CLI or Python API).

    Returns:
        TrainResult with best epoch, metrics, and saved weights path.
    """
    resume = getattr(args, "resume", None)
    hparams = load_train_hparams(resume) if resume else {}
    if resume and hparams:
        LOGGER.info(
            f"Loaded hparams from resume dir: model={hparams.get('model_name')}, dataset={hparams.get('dataset')}"
        )

    trainer_kwargs = trainer_kwargs_from_args(args, hparams)
    config = ReIDTrainConfig.from_flat_kwargs(**trainer_kwargs)
    trainer = ReIDTrainer.from_config(config)

    result = trainer.run()
    LOGGER.info(
        f"Training finished. Best weights: {result.weights_path}  mAP={result.best_mAP:.2%}  R1={result.best_rank1:.2%}"
    )
    return result
