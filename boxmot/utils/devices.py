"""Shared single-device selection that preserves process-wide GPU visibility."""

from __future__ import annotations

import torch


def normalize_device(value: str | torch.device) -> str:
    """Normalize one CPU, MPS, or logical CUDA selector without probing hardware.

    Numeric selectors refer to the GPUs already visible to the process, just
    like ``cuda:N``. Automatic selection belongs to the calling workflow;
    empty selectors and device lists are not supported here.
    """

    if isinstance(value, (str, torch.device)):
        device = str(value).strip().lower()
        if device in {"cpu", "mps"}:
            return device
        if device == "cuda":
            return "cuda:0"
        index = device.removeprefix("cuda:")
        if index.isdecimal():
            return f"cuda:{int(index)}"
    raise ValueError(
        f"Unsupported device {value!r}; expected a single device: cpu, mps, cuda, cuda:N, or N. "
        "GPU lists are not supported."
    )


def resolve_device(value: str | torch.device) -> torch.device:
    """Validate one device without remapping GPUs or selecting a global device.

    CUDA indices are relative to the existing ``CUDA_VISIBLE_DEVICES`` mask.
    Explicit accelerators must be available; this function never falls back
    to another device or sets the process's current CUDA device.
    """

    device = normalize_device(value)
    if device.startswith("cuda:"):
        index = int(device.removeprefix("cuda:"))
        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if index >= count:
            raise RuntimeError(
                f"Device {device} is unavailable; PyTorch reports {count} CUDA device(s). "
                "Select an available logical CUDA index or use device=cpu."
            )
    elif device == "mps":
        if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
            raise RuntimeError(
                "Device mps is unavailable in this PyTorch runtime. "
                "Use device=cpu or an MPS-enabled PyTorch build on a supported macOS host."
            )
    return torch.device(device)


__all__ = ("normalize_device", "resolve_device")
