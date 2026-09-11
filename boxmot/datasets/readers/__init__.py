"""Lazily resolved readers for declared data formats and published artifacts."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .artifacts import attach_masks, read_detection_batches, read_sample_records
    from .images import (
        NUMPY_IMAGE_EXTENSIONS,
        ImageDecodeError,
        probe_numpy_image_size,
        read_numpy_bgr_uint8,
        read_rgb_chw_uint8,
    )

__all__ = (
    "ImageDecodeError",
    "NUMPY_IMAGE_EXTENSIONS",
    "attach_masks",
    "probe_numpy_image_size",
    "read_detection_batches",
    "read_numpy_bgr_uint8",
    "read_rgb_chw_uint8",
    "read_sample_records",
)

_EXPORTS = {
    "ImageDecodeError": ("images", "ImageDecodeError"),
    "NUMPY_IMAGE_EXTENSIONS": ("images", "NUMPY_IMAGE_EXTENSIONS"),
    "attach_masks": ("artifacts", "attach_masks"),
    "probe_numpy_image_size": ("images", "probe_numpy_image_size"),
    "read_detection_batches": ("artifacts", "read_detection_batches"),
    "read_numpy_bgr_uint8": ("images", "read_numpy_bgr_uint8"),
    "read_rgb_chw_uint8": ("images", "read_rgb_chw_uint8"),
    "read_sample_records": ("artifacts", "read_sample_records"),
}


def __getattr__(name: str) -> Any:
    """Keep numeric frame discovery independent of pixels, tensors and parquet."""
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose public readers without importing optional processing libraries."""
    return sorted((*globals(), *__all__))
