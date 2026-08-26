"""Runtime, memory, normalization, and reproducibility support."""

from __future__ import annotations

import gc
import os
import random
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from boxmot.utils import logger as LOGGER


class _RuntimeMixin:
    def _memory_utilization(self) -> Optional[float]:
        """Return this process's accelerator-memory utilization when available."""
        if self.device.type == "cuda" and torch.cuda.is_available():
            total = torch.cuda.get_device_properties(self.device).total_memory
            return torch.cuda.memory_reserved(self.device) / total if total > 0 else None
        if self.device.type == "mps" and torch.backends.mps.is_available():
            total = torch.mps.recommended_max_memory()
            return torch.mps.driver_allocated_memory() / total if total > 0 else None
        return None

    def _clear_memory(self, *, force: bool = False, threshold: Optional[float] = None) -> bool:
        """Collect garbage and clear the accelerator cache on OOM or high utilization."""
        if self.device.type not in {"cuda", "mps"}:
            return False
        if not force:
            utilization = self._memory_utilization()
            if threshold is None or utilization is None or utilization < threshold:
                return False

        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        else:
            torch.mps.empty_cache()
        return True

    @staticmethod
    def _is_oom_error(exc: BaseException) -> bool:
        """Return whether an exception represents a CUDA/MPS out-of-memory failure."""
        return isinstance(exc, torch.OutOfMemoryError) or (
            isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
        )

    def _handle_oom(self, exc: BaseException, *optimizers) -> bool:
        """Release gradients and cached memory after an accelerator OOM."""
        if not self._is_oom_error(exc):
            return False
        for optimizer in optimizers:
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
        self._clear_memory(force=True)
        return True

    @staticmethod
    def _seed_everything(seed: int) -> None:
        """Seed every RNG used by model training and data augmentation."""
        seed = int(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed % 2**32)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)

    @staticmethod
    def _configure_determinism(enabled: bool) -> None:
        """Configure PyTorch backends to require or permit nondeterministic algorithms."""
        enabled = bool(enabled)
        if enabled:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = not enabled
            torch.backends.cudnn.deterministic = enabled
        torch.use_deterministic_algorithms(enabled)

    def _seed_training_epoch(self, epoch: int, loader: DataLoader) -> None:
        """Seed one epoch independently so fresh and resumed runs agree."""
        epoch_seed = self.seed + int(epoch)
        self._seed_everything(epoch_seed)
        self._train_generator.manual_seed(epoch_seed)
        dataset = getattr(loader, "dataset", None)
        target_setter = getattr(
            dataset,
            "set_anatomical_targets_enabled",
            None,
        )
        if callable(target_setter):
            target_setter(self._anatomical_training_active(epoch))
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        sampler = getattr(loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

    @staticmethod
    def _capture_rng_state() -> dict:
        """Capture process and accelerator RNG states for checkpoint resume."""
        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        if torch.backends.mps.is_available():
            state["mps"] = torch.mps.get_rng_state()
        return state

    @staticmethod
    def _restore_rng_state(state: Optional[dict]) -> None:
        """Restore RNG state saved in a training checkpoint."""
        if not state:
            LOGGER.warning("Checkpoint has no RNG state; the next epoch will use its configured seeded RNG stream")
            return
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"].cpu())
        if torch.cuda.is_available() and state.get("cuda") is not None:
            torch.cuda.set_rng_state_all([rng_state.cpu() for rng_state in state["cuda"]])
        if torch.backends.mps.is_available() and state.get("mps") is not None:
            torch.mps.set_rng_state(state["mps"].cpu())

    @staticmethod
    def _normalize_head_parts(head_parts) -> tuple[int, ...]:
        """Normalize CSL-TinyViT head part granularities from CLI/API inputs."""
        if isinstance(head_parts, str):
            parts = [part for part in head_parts.replace(";", ",").split(",") if part.strip()]
            return tuple(int(part) for part in parts)
        if isinstance(head_parts, int):
            return (int(head_parts),)
        return tuple(int(part) for part in head_parts)

    @staticmethod
    def _normalize_int_pair(value) -> tuple[int, int]:
        """Normalize integer-pair CLI/API inputs without deduplicating equal values."""
        if isinstance(value, int):
            return (int(value), int(value))
        if isinstance(value, str):
            tokens = [part for part in value.replace(";", ",").split(",") if part.strip()]
            if len(tokens) == 1:
                tokens = tokens * 2
            if len(tokens) != 2:
                raise ValueError(f"Expected one or two comma-separated integers, got {value!r}")
            return (int(tokens[0]), int(tokens[1]))
        values = tuple(int(part) for part in value)
        if len(values) == 1:
            return (values[0], values[0])
        if len(values) != 2:
            raise ValueError(f"Expected one or two integers, got {value!r}")
        return values

    @staticmethod
    def _normalize_post_fusion_mixer(mixer: str) -> str:
        """Normalize post-fusion local mixer aliases."""
        normalized = str(mixer).lower()
        if normalized in {"", "none", "off", "identity"}:
            return "none"
        if normalized in {"dwconv", "local", "dwconv5x3"}:
            return "dwconv"
        raise ValueError("post_fusion_mixer must be one of: none, dwconv")

    @staticmethod
    def _normalize_adapter_stages(stages) -> tuple[int, ...]:
        """Normalize CSL-TinyViT ReID adapter stage indices from CLI/API inputs."""
        if stages is None:
            return ()
        if isinstance(stages, str):
            if stages.lower() in {"", "none", "off"}:
                return ()
            parts = [part for part in stages.replace(";", ",").split(",") if part.strip()]
        elif isinstance(stages, int):
            parts = [stages]
        else:
            parts = list(stages)
        return tuple(dict.fromkeys(int(part) for part in parts))

    def _prepare_runtime(self) -> None:
        """Initialize deterministic process state and log effective runtime settings."""
        self._configure_determinism(self.deterministic)
        self._seed_everything(self.seed)
        LOGGER.info(f"Training reproducibility: seed={self.seed}, deterministic={self.deterministic}")
        if self.source_balance:
            LOGGER.info(
                f"Batch sizes: train={self.train_batch_size} "
                f"(source_balance={self.source_balance}), eval={self.eval_batch_size}"
            )
        else:
            LOGGER.info(
                f"Batch sizes: train={self.train_batch_size} (p={self.p} x k={self.k}), eval={self.eval_batch_size}"
            )
            if self.pk_steps_per_epoch or self.camera_aware_sampler:
                LOGGER.info(
                    "PK sampling: "
                    f"steps_per_epoch={self.pk_steps_per_epoch or 'identity-pass'}, "
                    f"camera_aware={self.camera_aware_sampler}"
                )
