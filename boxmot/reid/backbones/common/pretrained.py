# BoxMOT AGPL-3.0 license

from __future__ import annotations

import os
import warnings
from collections import OrderedDict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

StateDictKeyTransform = Callable[[str], str]
StateDictTensorTransform = Callable[[str, torch.Tensor], tuple[str, torch.Tensor]]


def _extract_state_dict(checkpoint: Any, checkpoint_key: str | None = None) -> Mapping[str, Any]:
    """Return a state dict from common checkpoint wrapper layouts."""
    if checkpoint_key is not None:
        if not isinstance(checkpoint, Mapping) or checkpoint_key not in checkpoint:
            raise KeyError(f"Checkpoint does not contain state-dict key: {checkpoint_key!r}")
        checkpoint = checkpoint[checkpoint_key]

    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Expected checkpoint mapping, got {type(checkpoint).__name__}")

    for key in ("state_dict", "model", "model_state_dict"):
        nested = checkpoint.get(key)
        if isinstance(nested, Mapping):
            return nested
    return checkpoint


def resolve_torch_cache_path(url: str, filename: str | None = None) -> Path:
    """Resolve the torch hub checkpoint cache path for a URL."""
    torch_home = Path(
        os.path.expanduser(
            os.getenv(
                "TORCH_HOME",
                os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "torch"),
            )
        )
    )
    return torch_home / "checkpoints" / (filename or url.rsplit("/", 1)[-1])


def load_torch_url(
    url: str,
    *,
    filename: str | None = None,
    map_location: str | torch.device = "cpu",
    weights_only: bool = False,
    logger=None,
) -> Any:
    """Download a URL into the torch cache if needed and load it."""
    cached = resolve_torch_cache_path(url, filename)
    cached.parent.mkdir(parents=True, exist_ok=True)
    if not cached.exists():
        if logger is not None:
            logger.info(f"Downloading pretrained weights from {url}")
        torch.hub.download_url_to_file(url, str(cached), progress=True)
    return torch.load(cached, map_location=map_location, weights_only=weights_only)


def load_hub_checkpoint(
    url: str,
    *,
    filename: str | None = None,
    checkpoint_key: str | None = None,
    map_location: str | torch.device = "cpu",
    weights_only: bool = False,
    logger=None,
) -> Mapping[str, Any]:
    """Load a checkpoint URL from the torch cache and return its state dict."""
    checkpoint = load_torch_url(
        url,
        filename=filename,
        map_location=map_location,
        weights_only=weights_only,
        logger=logger,
    )
    return _extract_state_dict(checkpoint, checkpoint_key=checkpoint_key)


def load_gdrive_url(
    url: str,
    *,
    filename: str,
    map_location: str | torch.device = "cpu",
    weights_only: bool = False,
    quiet: bool = False,
    logger=None,
) -> Any:
    """Download a Google Drive URL into the torch cache if needed and load it."""
    import gdown

    cached = resolve_torch_cache_path(url, filename)
    cached.parent.mkdir(parents=True, exist_ok=True)
    if not cached.exists():
        if logger is not None:
            logger.info(f"Downloading pretrained weights from {url}")
        gdown.download(url, str(cached), quiet=quiet)
    return torch.load(cached, map_location=map_location, weights_only=weights_only)


def load_gdrive_checkpoint(
    url: str,
    *,
    filename: str,
    checkpoint_key: str | None = None,
    map_location: str | torch.device = "cpu",
    weights_only: bool = False,
    quiet: bool = False,
    logger=None,
) -> Mapping[str, Any]:
    """Load a Google Drive checkpoint from the torch cache and return its state dict."""
    checkpoint = load_gdrive_url(
        url,
        filename=filename,
        map_location=map_location,
        weights_only=weights_only,
        quiet=quiet,
        logger=logger,
    )
    return _extract_state_dict(checkpoint, checkpoint_key=checkpoint_key)


def load_partial_state_dict(
    model: nn.Module,
    state_dict: Mapping[str, Any],
    *,
    strip_prefix: str | None = "module.",
    key_transform: StateDictKeyTransform | None = None,
    tensor_transform: StateDictTensorTransform | None = None,
    logger=None,
) -> tuple[list[str], list[str]]:
    """Load tensors that match by normalized key and shape.

    Returns:
        The matched model keys and skipped checkpoint keys.
    """
    model_state = model.state_dict()
    matched = OrderedDict()
    skipped: list[str] = []

    for original_key, value in state_dict.items():
        key = original_key
        if strip_prefix and key.startswith(strip_prefix):
            key = key[len(strip_prefix) :]
        if key_transform is not None:
            key = key_transform(key)
        if tensor_transform is not None and isinstance(value, torch.Tensor):
            key, value = tensor_transform(key, value)

        if isinstance(value, torch.Tensor) and key in model_state and model_state[key].shape == value.shape:
            matched[key] = value
        else:
            skipped.append(original_key)

    model_state.update(matched)
    model.load_state_dict(model_state)

    if logger is not None:
        logger.info(f"Loaded {len(matched)} tensors, skipped {len(skipped)} tensors")

    return list(matched), skipped


def load_url_pretrained(
    model: nn.Module,
    url: str,
    *,
    filename: str | None = None,
    checkpoint_key: str | None = None,
    strip_prefix: str | None = "module.",
    key_transform: StateDictKeyTransform | None = None,
    tensor_transform: StateDictTensorTransform | None = None,
    logger=None,
) -> tuple[list[str], list[str]]:
    """Load matching pretrained tensors from a URL-backed torch checkpoint."""
    state_dict = load_hub_checkpoint(
        url,
        filename=filename,
        checkpoint_key=checkpoint_key,
        logger=logger,
        weights_only=False,
    )
    return load_partial_state_dict(
        model,
        state_dict,
        strip_prefix=strip_prefix,
        key_transform=key_transform,
        tensor_transform=tensor_transform,
        logger=logger,
    )


def log_pretrained_result(
    source: str,
    matched: list[str],
    skipped: list[str],
    *,
    logger=None,
    empty_warning: str | None = None,
) -> None:
    """Log a standardized pretrained-load outcome."""
    if not matched:
        warnings.warn(
            empty_warning
            or f'Pretrained weights from "{source}" cannot be loaded, '
            "please check the key names manually (** ignored and continue **)",
            stacklevel=2,
        )
        return

    if logger is not None:
        logger.info(f'Successfully loaded pretrained weights from "{source}"')
        if skipped:
            logger.debug(f"Skipped pretrained layers: {skipped}")


def warn_manual_pretrained_download(url: str, *, source: str = "imagenet pretrained weights") -> None:
    """Warn that a pretrained checkpoint requires manual download."""
    warnings.warn(f"The {source} need to be manually downloaded from {url}", stacklevel=2)
