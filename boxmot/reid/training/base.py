# BoxMOT AGPL-3.0 license

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from boxmot.utils import logger as LOGGER


class BaseTrainer(ABC):
    """A base class for creating trainers.

    This class provides the foundation for training ReID models, handling the
    training loop, validation, checkpointing, and shared training utilities
    through template-method hooks implemented by concrete trainers.
    """

    def run(self) -> Any:
        """Execute the full training pipeline."""
        run_started_at = time.monotonic()
        self._prepare_runtime()
        data = self._build_dataset_bundle()
        models = self._build_model_bundle(data.num_classes)
        loaders = self._build_loader_bundle(data)
        losses = self._build_loss_bundle(models, data.num_classes)
        optimization = self._build_optimization_bundle(models, losses)
        state = self._restore_if_needed(models, loaders, losses, optimization)

        save_dir = self._resolve_save_dir()
        LOGGER.info(f"Saving results to {save_dir}")
        self._write_hparams(save_dir, data, models, losses)
        return self._fit(
            save_dir=save_dir,
            data=data,
            models=models,
            loaders=loaders,
            losses=losses,
            optimization=optimization,
            state=state,
            run_started_at=run_started_at,
        )

    def _resolve_save_dir(self) -> Path:
        """Return the active output directory for new or resumed training."""
        resume = getattr(self, "resume", None)
        if resume:
            resume_path = Path(resume)
            return resume_path if resume_path.is_dir() else resume_path.parent
        return self._make_save_dir()

    @abstractmethod
    def _prepare_runtime(self) -> None:
        """Prepare random seeds, deterministic settings, and runtime state."""

    @abstractmethod
    def _build_dataset_bundle(self) -> Any:
        """Build the training dataset bundle."""

    @abstractmethod
    def _build_model_bundle(self, num_classes: int) -> Any:
        """Build live and validation model references."""

    @abstractmethod
    def _build_loader_bundle(self, data: Any) -> Any:
        """Build training and validation loaders."""

    @abstractmethod
    def _build_loss_bundle(self, model: Any, num_classes: int) -> Any:
        """Build all loss modules needed for training."""

    @abstractmethod
    def _build_optimization_bundle(self, model: Any, losses: Any) -> Any:
        """Build optimizers, schedulers, and clipping policy."""

    @abstractmethod
    def _restore_if_needed(self, models: Any, loaders: Any, losses: Any, optimization: Any) -> Any:
        """Restore a resumable training state when requested."""

    @abstractmethod
    def _write_hparams(self, save_dir: Path, data: Any, models: Any, losses: Any) -> None:
        """Persist resolved hyperparameters for the run."""

    @abstractmethod
    def _fit(
        self,
        *,
        save_dir: Path,
        data: Any,
        models: Any,
        loaders: Any,
        losses: Any,
        optimization: Any,
        state: Any,
        run_started_at: float,
    ) -> Any:
        """Run the concrete training loop."""

    @abstractmethod
    def _make_save_dir(self) -> Path:
        """Create a new output directory for a fresh training run."""
