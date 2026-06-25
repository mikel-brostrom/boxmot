"""Checkpoint persistence for ReID training."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.nn as nn


class CheckpointManager:
    """Persist distinct resumable and inference checkpoints."""

    def __init__(
        self,
        *,
        metadata_factory: Callable[[nn.Module], dict[str, Any]],
        rng_state_factory: Callable[[], dict[str, Any]],
        classifier_loss: str,
    ):
        self.metadata_factory = metadata_factory
        self.rng_state_factory = rng_state_factory
        self.classifier_loss = classifier_loss

    def save_last(
        self,
        path: Path,
        *,
        model: nn.Module,
        epoch: int,
        val: Optional[Any],
        optimizer,
        optimizer_center,
        criterion_center,
        criterion_classifier,
        ema_model: Optional[nn.Module],
        best_mAP: float,
    ) -> None:
        """Save the live model and optimizer state required for exact resume."""
        self._save(
            path,
            model=model,
            epoch=epoch,
            val=val,
            checkpoint_type="last",
            resumable=True,
            optimizer=optimizer,
            optimizer_center=optimizer_center,
            criterion_center=criterion_center,
            criterion_classifier=criterion_classifier,
            ema_model=ema_model,
            best_mAP=best_mAP,
        )

    def save_best(
        self,
        path: Path,
        *,
        model: nn.Module,
        epoch: int,
        val: Any,
        criterion_center,
        criterion_classifier,
        best_mAP: float,
    ) -> None:
        """Save inference weights without claiming optimizer-resume compatibility."""
        self._save(
            path,
            model=model,
            epoch=epoch,
            val=val,
            checkpoint_type="best",
            resumable=False,
            criterion_center=criterion_center,
            criterion_classifier=criterion_classifier,
            best_mAP=best_mAP,
        )

    def save(
        self,
        path: Path,
        *,
        model: nn.Module,
        epoch: int,
        val: Optional[Any],
        checkpoint_type: str = "manual",
        resumable: bool = False,
        optimizer=None,
        optimizer_center=None,
        criterion_center=None,
        criterion_classifier=None,
        ema_model: Optional[nn.Module] = None,
        best_mAP: float = 0.0,
    ) -> None:
        """Compatibility entry point for explicit checkpoint saves."""
        self._save(
            path,
            model=model,
            epoch=epoch,
            val=val,
            checkpoint_type=checkpoint_type,
            resumable=resumable,
            optimizer=optimizer,
            optimizer_center=optimizer_center,
            criterion_center=criterion_center,
            criterion_classifier=criterion_classifier,
            ema_model=ema_model,
            best_mAP=best_mAP,
        )

    def _save(
        self,
        path: Path,
        *,
        model: nn.Module,
        epoch: int,
        val: Optional[Any],
        checkpoint_type: str,
        resumable: bool,
        optimizer=None,
        optimizer_center=None,
        criterion_center=None,
        criterion_classifier=None,
        ema_model: Optional[nn.Module] = None,
        best_mAP: float = 0.0,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            **self.metadata_factory(model),
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "checkpoint_type": checkpoint_type,
            "resumable": resumable,
            "best_mAP": best_mAP,
            "rng_state": self.rng_state_factory(),
        }
        if val is not None:
            state["mAP"] = val.mAP
            state["rank1"] = val.rank1
        if optimizer is not None:
            state["optimizer"] = optimizer.state_dict()
        if optimizer_center is not None:
            state["optimizer_center"] = optimizer_center.state_dict()
        if criterion_center is not None:
            state["center_loss_state_dict"] = criterion_center.state_dict()
        if criterion_classifier is not None and self.classifier_loss != "ce":
            state["classifier_loss_state_dict"] = criterion_classifier.state_dict()
        if ema_model is not None:
            state["ema_state_dict"] = ema_model.state_dict()
        torch.save(state, path)
