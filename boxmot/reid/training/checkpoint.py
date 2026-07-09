"""Checkpoint persistence for ReID training."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.nn as nn


FP16_MAX = torch.finfo(torch.float16).max
TRAIN_ONLY_STATE_DICT_PREFIXES = ("classifier.",)
TRAIN_ONLY_STATE_DICT_INFIXES = (".classifier.",)


def _is_train_only_model_key(key: str) -> bool:
    """Return True for model parameters needed only by ID classification training."""
    return key.startswith(TRAIN_ONLY_STATE_DICT_PREFIXES) or any(
        marker in key for marker in TRAIN_ONLY_STATE_DICT_INFIXES
    )


def _compact_floating_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return a finite FP16 copy for compact checkpoint serialization."""
    if not torch.is_floating_point(tensor):
        return tensor

    compact = tensor.detach().clone()
    compact = torch.nan_to_num(compact, nan=0.0, posinf=FP16_MAX, neginf=-FP16_MAX)
    compact = compact.clamp(min=-FP16_MAX, max=FP16_MAX)
    return compact.half()


def _compact_floating_tensors(value: Any) -> Any:
    """Recursively cast floating tensors to FP16 while preserving containers."""
    if torch.is_tensor(value):
        return _compact_floating_tensor(value)
    if isinstance(value, dict):
        return value.__class__((key, _compact_floating_tensors(item)) for key, item in value.items())
    if isinstance(value, list):
        return [_compact_floating_tensors(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_compact_floating_tensors(item) for item in value)
    return value


def _compact_model_state_dict(model: nn.Module, *, resumable: bool) -> dict[str, Any]:
    """Return compact model weights, dropping train-only heads for inference checkpoints."""
    state_dict = model.state_dict()
    if not resumable:
        state_dict = state_dict.__class__(
            (key, value) for key, value in state_dict.items() if not _is_train_only_model_key(key)
        )
    return _compact_floating_tensors(state_dict)


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
        """Save the live model and optimizer state required for resume."""
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
        """Save compact inference weights without claiming optimizer-resume compatibility."""
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
            "state_dict": _compact_model_state_dict(model, resumable=resumable),
            "epoch": epoch,
            "checkpoint_type": checkpoint_type,
            "resumable": resumable,
            "best_mAP": best_mAP,
            "checkpoint_precision": "float16",
        }
        if resumable:
            state["rng_state"] = self.rng_state_factory()
        if val is not None:
            state["mAP"] = val.mAP
            state["rank1"] = val.rank1
        if optimizer is not None:
            state["optimizer"] = _compact_floating_tensors(optimizer.state_dict())
        if optimizer_center is not None:
            state["optimizer_center"] = _compact_floating_tensors(optimizer_center.state_dict())
        if resumable and criterion_center is not None:
            state["center_loss_state_dict"] = _compact_floating_tensors(criterion_center.state_dict())
        if resumable and criterion_classifier is not None and self.classifier_loss != "ce":
            state["classifier_loss_state_dict"] = _compact_floating_tensors(criterion_classifier.state_dict())
        if resumable and ema_model is not None:
            state["ema_state_dict"] = _compact_floating_tensors(ema_model.state_dict())
        torch.save(state, path)
