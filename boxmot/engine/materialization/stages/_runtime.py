"""Runtime cleanup shared by accelerator-backed materialization stages."""

from __future__ import annotations

import gc

import torch


def release_accelerator_memory(device: str) -> None:
    """Collect released models and return cached accelerator blocks to the OS.

    MPS uses unified system memory, so retaining its caching allocator between
    perception and wide Parquet compaction can force finalization into swap.
    CUDA benefits from the same lifecycle boundary without changing results.
    """

    if device != "mps" and device != "cuda" and not device.startswith("cuda:"):
        return

    gc.collect()
    if (device == "cuda" or device.startswith("cuda:")) and torch.cuda.is_available():
        torch.cuda.empty_cache()
    if device == "mps" and torch.backends.mps.is_available():
        torch.mps.empty_cache()


__all__ = ("release_accelerator_memory",)
