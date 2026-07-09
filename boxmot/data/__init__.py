from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from boxmot.data.cache import AppendableNpyWriter
    from boxmot.data.dataset import MOTDataset, MOTSequence, compute_fps_mask, read_seq_fps
    from boxmot.data.loaders import IMAGE_EXTS, MANIFEST_EXTS, VIDEO_EXTS, iter_source

_EXPORTS = {
    "AppendableNpyWriter": ("boxmot.data.cache", "AppendableNpyWriter"),
    "IMAGE_EXTS": ("boxmot.data.loaders", "IMAGE_EXTS"),
    "MANIFEST_EXTS": ("boxmot.data.loaders", "MANIFEST_EXTS"),
    "MOTDataset": ("boxmot.data.dataset", "MOTDataset"),
    "MOTSequence": ("boxmot.data.dataset", "MOTSequence"),
    "VIDEO_EXTS": ("boxmot.data.loaders", "VIDEO_EXTS"),
    "compute_fps_mask": ("boxmot.data.dataset", "compute_fps_mask"),
    "iter_source": ("boxmot.data.loaders", "iter_source"),
    "read_seq_fps": ("boxmot.data.dataset", "read_seq_fps"),
}

__all__ = tuple(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)

    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
